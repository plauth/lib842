// Compression - decompression benchmark for non-GPU implementations,
// based on multithreading using standard C++ threads
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
#include "compdecomp_driver.h"

#if defined(USEAIX)
#include <sys/types.h>
#include <sys/vminfo.h>
#define ALIGNMENT 4096
#define lib842_decompress(in, ilen, out, olen) accel_decompress(in, ilen, out, olen, 0)
#define lib842_compress(in, ilen, out, olen) accel_compress(in, ilen, out, olen, 0)
#elif defined(USEHW)
#include "hw842.h"
#define ALIGNMENT 0
#define lib842_decompress hw842_decompress
#define lib842_compress hw842_compress
#elif defined(USEOPTSW)
#include "sw842.h"
#define ALIGNMENT 0
#define lib842_decompress optsw842_decompress
#define lib842_compress optsw842_compress
#else
#include "sw842.h"
#define ALIGNMENT 0
#define lib842_decompress sw842_decompress
#define lib842_compress sw842_compress
#endif

//#define CHUNK_SIZE ((size_t)32768)
//#define CHUNK_SIZE ((size_t)1024)
#define CHUNK_SIZE ((size_t)4096)

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

bool compress_benchmark_core(const uint8_t *in, size_t ilen,
			     uint8_t *out, size_t *olen,
			     uint8_t *decompressed, size_t *dlen,
			     long long *time_comp,
			     long long *time_condense,
			     long long *time_decomp) {
	// -----
	// SETUP
	// -----
	bool ret = false;
	size_t num_threads = determine_num_threads();

	size_t num_chunks = ilen / CHUNK_SIZE;
	std::vector<size_t> compressed_chunk_sizes(num_chunks);
	std::vector<size_t> decompressed_chunk_sizes(num_chunks);

	// -----------
	// COMPRESSION
	// -----------
	unsigned int compthreads_ready = 0;
	bool compthread_exit = false;
	std::mutex compthread_mutex;
	std::condition_variable compthread_trigger;
	std::queue<size_t> compthread_queue;
	std::atomic<bool> compthread_error(false);

	std::vector<std::thread> compthreads(num_threads);
	for (size_t i = 0; i < compthreads.size(); i++) {
		compthreads[i] = std::thread([&compthreads_ready,
					      &compthread_exit,
					      &compthread_mutex,
					      &compthread_trigger,
					      &compthread_queue,
					      &compthread_error,
					      &in, &out,
					      &compressed_chunk_sizes] {
			// Notify the owner we're ready
			{
				std::unique_lock<std::mutex> lock(compthread_mutex);
				compthreads_ready++;
				compthread_trigger.notify_all();
			}

			while (true) {
				// Wait for work on the queue, or an exit message
				std::unique_lock<std::mutex> lock(compthread_mutex);
				compthread_trigger.wait(lock, [&compthread_queue, &compthread_exit] {
					return !compthread_queue.empty() || compthread_exit;
				});
				if (compthread_queue.empty() && compthread_exit)
					break;
				size_t chunk_num = compthread_queue.front();
				compthread_queue.pop();


				// Do the actual work
				size_t chunk_olen = CHUNK_SIZE * 2;
				const uint8_t *chunk_in = in + (CHUNK_SIZE * chunk_num);
				uint8_t *chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);

				int err = lib842_compress(chunk_in, CHUNK_SIZE, chunk_out,
							  &chunk_olen);
				if (err < 0 && !compthread_error.exchange(true)) {
					fprintf(stderr, "FAIL: Error during compression (%d): %s\n",
					        -err, strerror(-err));
				}
				compressed_chunk_sizes[chunk_num] = chunk_olen;
			}
		});
	}

	// Wait until the threads are actually running, as to
	// not to include thread spawning overhead in the timing
	{
		std::unique_lock<std::mutex> lock(compthread_mutex);
		compthread_trigger.wait(lock, [&compthreads_ready, num_threads] {
			return compthreads_ready == num_threads;
		});
	}

	long long timestart_comp = timestamp();

	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
		std::unique_lock<std::mutex> lock(compthread_mutex);
		compthread_queue.push(chunk_num);
		compthread_trigger.notify_one();
	}
	compthread_exit = true;
	compthread_trigger.notify_all();
	for (size_t i = 0; i < compthreads.size(); i++)
		compthreads[i].join();

	if (compthread_error)
		return false;

	*time_comp = timestamp() - timestart_comp;

	*olen = 0;
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++)
		*olen += compressed_chunk_sizes[chunk_num];

	// ------------
	// CONDENSATION
	// ------------
	*time_condense = -1;

	// -------------
	// DECOMPRESSION
	// -------------
	unsigned int decompthreads_ready = 0;
	bool decompthread_exit = false;
	std::mutex decompthread_mutex;
	std::condition_variable decompthread_trigger;
	std::queue<size_t> decompthread_queue;
	std::atomic<bool> decompthread_error(false);

	std::vector<std::thread> decompthreads(num_threads);
	for (size_t i = 0; i < decompthreads.size(); i++) {
		decompthreads[i] = std::thread([&decompthreads_ready,
						&decompthread_exit,
						&decompthread_mutex,
						&decompthread_trigger,
						&decompthread_queue,
						&decompthread_error,
						&out, &decompressed,
						&compressed_chunk_sizes,
						&decompressed_chunk_sizes] {
			// Notify the owner we're ready
			{
				std::unique_lock<std::mutex> lock(decompthread_mutex);
				decompthreads_ready++;
				decompthread_trigger.notify_all();
			}

			while (true) {
				// Wait for work on the queue, or an exit message
				std::unique_lock<std::mutex> lock(decompthread_mutex);
				decompthread_trigger.wait(lock, [&decompthread_queue, &decompthread_exit] {
					return !decompthread_queue.empty() || decompthread_exit;
				});
				if (decompthread_queue.empty() && decompthread_exit)
					break;
				size_t chunk_num = decompthread_queue.front();
				decompthread_queue.pop();

				// Do the actual work
				size_t chunk_dlen = CHUNK_SIZE;
				uint8_t *chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);
				uint8_t *chunk_decomp = decompressed + (CHUNK_SIZE * chunk_num);

				int err = lib842_decompress(chunk_out,
							    compressed_chunk_sizes[chunk_num],
							    chunk_decomp, &chunk_dlen);
				if (err < 0 && !decompthread_error.exchange(true)) {
					fprintf(stderr, "FAIL: Error during decompression (%d): %s\n",
					        -err, strerror(-err));
				}
				decompressed_chunk_sizes[chunk_num] = chunk_dlen;
			}
		});
	}

	// Wait until the threads are actually running, as to
	// not to include thread spawning overhead in the timing
	{
		std::unique_lock<std::mutex> lock(decompthread_mutex);
		decompthread_trigger.wait(lock, [&decompthreads_ready, num_threads] {
			return decompthreads_ready == num_threads;
		});
	}

	long long timestart_decomp = timestamp();

	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
		std::unique_lock<std::mutex> lock(decompthread_mutex);
		decompthread_queue.push(chunk_num);
		decompthread_trigger.notify_one();
	}
	decompthread_exit = true;
	decompthread_trigger.notify_all();
	for (size_t i = 0; i < decompthreads.size(); i++)
		decompthreads[i].join();

	if (decompthread_error)
		return false;

	*time_decomp = timestamp() - timestart_decomp;

	*dlen = 0;
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++)
		*dlen += decompressed_chunk_sizes[chunk_num];

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
	if (err < 0) {
		fprintf(stderr, "Error during compression (%d): %s\n",
		        -err, strerror(-err));
		return false;
	}

	err = lib842_decompress(out, *olen, decompressed, dlen);
	if (err < 0) {
		fprintf(stderr, "Error during decompression (%d): %s\n",
		        -err, strerror(-err));
		return false;
	}

	return true;
}

int main(int argc, const char *argv[])
{
	return compdecomp(argc > 1 ? argv[1] : NULL, CHUNK_SIZE, ALIGNMENT);
}
