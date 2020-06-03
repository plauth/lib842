// Compression - decompression benchmark for non-GPU implementations,
// based on multithreading using standard C++ threads
#include "compdecomp_driver.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <sstream>

#if defined(USEAIX)
#include <lib842/aix.h>
#define lib842impl aix842_implementation
#elif defined(USEHW)
#include <lib842/hw.h>
#define lib842impl hw842_implementation
#elif defined(USEOPTSW)
#include <lib842/sw.h>
#define lib842impl optsw842_implementation
#else
#include <lib842/sw.h>
#define lib842impl sw842_implementation
#endif

#include <lib842/stream/comp.h>
#include <lib842/stream/decomp.h>
#include <lib842/detail/latch.h>

// Wraps an std::ostream and gathers multiple individual writes into an
// individual, thread-safe atomic write, to avoid interleaving of output
// between multiple threads. Similar to C++20's std::ostreambuf.
// (This class itself not thread safe. Do not share instances between threads)
class osyncstream : public std::ostream {
public:
	explicit osyncstream(std::ostream &stream)
		: std::ostream(&_buffer), _buffer(stream) { }
private:
	class osyncstreambuf: public std::stringbuf {
	public:
		explicit osyncstreambuf(std::ostream &stream) : _stream(stream) { }

		int sync() override {
			std::lock_guard<std::mutex> lock(_mutex);
			_stream << str();
			str("");
			return 0;
		}
	private:
		std::ostream &_stream;
		static std::mutex _mutex;
	};

	osyncstreambuf _buffer;
};
std::mutex osyncstream::osyncstreambuf::_mutex;

// Helper for std::unique_ptr for releasing C pointers with std::free()
struct free_ptr {
	void operator()(uint8_t *p) const {
		std::free(p);
	}
};

static unsigned int determine_num_threads() {
	// Configuration for the number of threads to use for compression or decompression
	const char *env_value = std::getenv("COMPDECOMP_NUM_THREADS");
	if (env_value != nullptr && std::atoi(env_value) > 0) {
		return (unsigned int)std::atoi(env_value);
	}

	// If the value is not specified (or invalid),
	// the hardware concurrency level (~= number of logical cores) is used
	static unsigned int hardware_concurrency = std::thread::hardware_concurrency();
	if (hardware_concurrency == 0) {
		fprintf(stderr, "std::thread::hardware_concurrency() returned 0, using 1 thread\n");
		return 1;
	}

	return hardware_concurrency;
}


static lib842::stream::thread_policy determine_thread_policy() {
	return std::getenv("COMPDECOMP_NUMA_SPREAD") == nullptr
		? lib842::stream::thread_policy::use_defaults
		: lib842::stream::thread_policy::spread_threads_among_numa_nodes;
}

bool compress_benchmark_core(const uint8_t *in, size_t ilen,
			     size_t *olen,
			     long long *time_comp,
			     long long *time_condense,
			     long long *time_decomp) {
	std::unique_ptr<uint8_t, free_ptr> decompressed(
		static_cast<uint8_t *>(allocate_aligned(ilen, lib842impl.alignment)));
	if (decompressed.get() == nullptr) {
		fprintf(stderr, "FAIL: decompressed = allocate_aligned(...) failed!\n");
		return false;
	}
	std::memset(decompressed.get(), 0, ilen);

#if 0
	{ // TODOXXX: Test to see if pre-paging memory has an influence on performance
		std::vector<std::unique_ptr<uint8_t, free_ptr>> outp;
		for (size_t i = 0; i < ilen / lib842::stream::CHUNK_SIZE; i++) {
			std::unique_ptr<uint8_t, free_ptr> out(
				static_cast<uint8_t *>(allocate_aligned(lib842::stream::CHUNK_SIZE * 2, lib842impl.alignment)));
			if (out.get() == nullptr) {
				fprintf(stderr, "FAIL: out = allocate_aligned(...) failed!\n");
				return false;
			}
			memset(out.get(), 0, ilen * 2);
		}
	}
#endif

	// -----
	// SETUP
	// -----
	auto num_threads = determine_num_threads();
	auto thread_policy = determine_thread_policy();

	auto get_log_debug = []() -> std::ostream& {
		static thread_local osyncstream log_debug(std::cout);
		return log_debug;
	};
	auto get_log_error = []() -> std::ostream& {
		static thread_local osyncstream log_error(std::cerr);
		return log_error;
	};

	// -----------
	// COMPRESSION
	// -----------
	// TODOXXX: Why is the performance on the first run so horrible?
	//          Is it due to NUMA effects? Why does it not happen with OpenMP?
	//          It could also be because a lot of new memory needs to be paged in?
	//          (In the OpenMP version, the memset will do this!)
	std::vector<lib842::stream::DataCompressionStream::compress_block> comp_blocks;
	{
		//for (int i = 0; i < 2; i++) { comp_blocks.clear();
		std::mutex comp_blocks_mutex;
		bool comp_error = false;
		lib842::stream::DataCompressionStream cstream(
			lib842impl.compress, num_threads, thread_policy,
			get_log_error, get_log_debug);
		cstream.wait_until_ready();

		long long timestart_comp = timestamp();
		cstream.start(in, ilen, false, [&comp_blocks, &comp_blocks_mutex, &comp_error]
			(lib842::stream::DataCompressionStream::compress_block &&cblock) {
			std::lock_guard<std::mutex> lock(comp_blocks_mutex);
			if (cblock.source_offset == SIZE_MAX)
				comp_error = true;
			if (!comp_error)
				comp_blocks.push_back(std::move(cblock));
		});

		lib842::detail::latch comp_finished(1);
		comp_error = false;
		cstream.finalize(false, [&comp_finished,
					 &comp_error](bool success) {
			comp_error |= !success;
			comp_finished.count_down();
		});

		comp_finished.wait();
		if (comp_error)
			return false;

		*time_comp = timestamp() - timestart_comp;
		//printf("TIME COMP%i: %lli\n", i, *time_comp); }
	}

	*olen = 0;
	for (const auto &cblock : comp_blocks) {
		for (auto size : cblock.sizes)
			*olen += size;
	}

	// ------------
	// CONDENSATION
	// ------------
	*time_condense = -1;

	// -------------
	// DECOMPRESSION
	// -------------
	{
		lib842::stream::DataDecompressionStream dstream(
			lib842impl.decompress, num_threads, thread_policy,
			get_log_error, get_log_debug);

		dstream.wait_until_ready();

		long long timestart_decomp = timestamp();

		dstream.start();
		for (const auto &cblock : comp_blocks) {
			lib842::stream::DataDecompressionStream::decompress_block dblock;
			bool any_compressed = false;
			for (size_t i = 0; i < lib842::stream::NUM_CHUNKS_PER_BLOCK; i++) {
				auto dest = decompressed.get() + cblock.source_offset + i * lib842::stream::CHUNK_SIZE;
				if (cblock.sizes[i] <= lib842::stream::COMPRESSIBLE_THRESHOLD) {
					dblock.chunks[i] = lib842::stream::DataDecompressionStream::decompress_chunk(
						cblock.datas[i],
						cblock.sizes[i],
						dest
					);
					any_compressed = true;
				} else {
					// TODOXXX: Should this be done multi thread???
					//          Or maybe done separately and not even included in the
					//          timing, since that's usually handled by the network?
					memcpy(dest, cblock.datas[i], lib842::stream::CHUNK_SIZE);
				}
			}

			if (any_compressed && !dstream.push_block(std::move(dblock)))
				break;
		}

		lib842::detail::latch decomp_finished(1);
		bool decomp_error = false;
		dstream.finalize(false, [&decomp_finished,
					 &decomp_error](bool success) {
			decomp_error = !success;
			decomp_finished.count_down();
		});

		decomp_finished.wait();
		if (decomp_error)
			return false;

		*time_decomp = timestamp() - timestart_decomp;
	}

	// ----------
	// VALIDATION
	// ----------
	if (ilen != comp_blocks.size() * lib842::stream::BLOCK_SIZE ||
	    memcmp(in, decompressed.get(), ilen) != 0) {
		fprintf(stderr,
			"FAIL: Decompressed data differs from the original input data.\n");
		return false;
	}

	return true;
}

bool simple_test_core(const uint8_t *in, size_t ilen,
		      uint8_t *out, size_t *olen,
		      uint8_t *decompressed, size_t *dlen)
{
	int err;

	err = lib842impl.compress(in, ilen, out, olen);
	if (err != 0) {
		fprintf(stderr, "Error during compression (%d): %s\n",
			-err, strerror(-err));
		return false;
	}

	err = lib842impl.decompress(out, *olen, decompressed, dlen);
	if (err != 0) {
		fprintf(stderr, "Error during decompression (%d): %s\n",
			-err, strerror(-err));
		return false;
	}

	return true;
}

int main(int argc, const char *argv[])
{
	return compdecomp(argc > 1 ? argv[1] : nullptr,
		lib842::stream::BLOCK_SIZE, lib842impl.alignment);
}
