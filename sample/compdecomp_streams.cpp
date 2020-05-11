// Compression - decompression benchmark for non-GPU implementations,
// based on multithreading using standard C++ threads
#include "compdecomp_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>

#if defined(USEAIX)
#include <sys/types.h>
#include <sys/vminfo.h>
#define ALIGNMENT 4096
static int lib842_decompress(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen) {
	return accel_decompress(in, ilen, out, olen, 0);
}
static int lib842_compress(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen) {
	return accel_compress(in, ilen, out, olen, 0);
}
#elif defined(USEHW)
#include <lib842/hw.h>
#define ALIGNMENT 0
#define lib842_decompress hw842_decompress
#define lib842_compress hw842_compress
#elif defined(USEOPTSW)
#include <lib842/sw.h>
#define ALIGNMENT 0
#define lib842_decompress optsw842_decompress
#define lib842_compress optsw842_compress
#else
#include <lib842/sw.h>
#define ALIGNMENT 0
#define lib842_decompress sw842_decompress
#define lib842_compress sw842_compress
#endif

#include <lib842/stream/comp.h>
#include <lib842/stream/decomp.h>
#include <lib842/detail/latch.h>

static unsigned int determine_num_threads()
{
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
	return std::getenv("COMPDECOMP_NUMA_SPREAD") != nullptr
		? lib842::stream::thread_policy::use_defaults
		: lib842::stream::thread_policy::spread_threads_among_numa_nodes;
}

bool compress_benchmark_core(const uint8_t *in, size_t ilen,
			     uint8_t *out, size_t *olen,
			     uint8_t *decompressed, size_t *dlen,
			     long long *time_comp,
			     long long *time_condense,
			     long long *time_decomp) {
	// -----
	// SETUP
	// -----
	auto num_threads = determine_num_threads();
	auto thread_policy = determine_thread_policy();

	// -----------
	// COMPRESSION
	// -----------
	// TODOXXX: Why is the performance on the first run so horrible?
	//          Is it due to NUMA effects? Why does it not happen with OpenMP?
	std::vector<lib842::stream::DataCompressionStream::compress_block> comp_blocks;
	std::mutex comp_blocks_mutex;
	bool comp_error = false;

	/* for (int i = 0; i < 2; i++) */ {
	lib842::stream::DataCompressionStream cstream(
		lib842_compress, num_threads, thread_policy,
		[]() -> std::ostream& { return std::cerr; },
		[]() -> std::ostream& { return std::cout; });
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
	cstream.finish(false);
	if (comp_error)
		return false;

	*time_comp = timestamp() - timestart_comp;

	*olen = 0;
	for (const auto &cblock : comp_blocks) {
		for (auto size : cblock.sizes)
			*olen += size;
	}

	// printf("TIME COMP%i: %lli\n", i, *time_comp);
	}

	// ------------
	// CONDENSATION
	// ------------
	*time_condense = -1;

	// -------------
	// DECOMPRESSION
	// -------------
	lib842::stream::DataDecompressionStream dstream(
		lib842_decompress, num_threads, thread_policy,
		[]() -> std::ostream& { return std::cerr; },
		[]() -> std::ostream& { return std::cout; });

	dstream.wait_until_ready();

	long long timestart_decomp = timestamp();

	dstream.start();
	for (const auto &cblock : comp_blocks) {
		lib842::stream::DataDecompressionStream::decompress_block dblock;
		bool any_compressed = false;
		for (size_t i = 0; i < lib842::stream::NUM_CHUNKS_PER_NETWORK_BLOCK; i++) {
			auto dest = decompressed + cblock.source_offset + i * lib842::stream::COMPR842_CHUNK_SIZE;
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
				memcpy(dest, cblock.datas[i], lib842::stream::COMPR842_CHUNK_SIZE);
			}
		}

		if (any_compressed && !dstream.push_block(std::move(dblock)))
			return false;
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

	*dlen = comp_blocks.size() * lib842::stream::NETWORK_BLOCK_SIZE;

	// ----------
	// VALIDATION
	// ----------
	if (ilen != *dlen || memcmp(in, decompressed, ilen) != 0) {
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

	err = lib842_compress(in, ilen, out, olen);
	if (err != 0) {
		fprintf(stderr, "Error during compression (%d): %s\n",
			-err, strerror(-err));
		return false;
	}

	err = lib842_decompress(out, *olen, decompressed, dlen);
	if (err != 0) {
		fprintf(stderr, "Error during decompression (%d): %s\n",
			-err, strerror(-err));
		return false;
	}

	return true;
}

int main(int argc, const char *argv[])
{
	return compdecomp(argc > 1 ? argv[1] : NULL,
		lib842::stream::NETWORK_BLOCK_SIZE, ALIGNMENT);
}
