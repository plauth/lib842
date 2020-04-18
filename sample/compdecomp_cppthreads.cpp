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

// If set, threads will be distributed evenly among available NUMA nodes
// This should provide an effect like OpenMP's OMP_PLACES=sockets OMP_PROC_BIND=spread
#define SPREAD_THREADS_AMONG_NUMA_NODES

#ifdef SPREAD_THREADS_AMONG_NUMA_NODES
#include <pthread.h>
#include <numa.h>
#include <sys/sysinfo.h>
#include <sstream>
#endif

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

#ifdef SPREAD_THREADS_AMONG_NUMA_NODES

static std::string cpu_set_to_string(cpu_set_t *cpuset)
{
	std::ostringstream cpuset_ss;
	for (int c = 0; c < CPU_SETSIZE; c++) {
		if (!CPU_ISSET(c, cpuset))
			continue;

		int cs = c;
		while (c + 1 < CPU_SETSIZE && CPU_ISSET(c + 1, cpuset))
			c++;

		if (cs == c)
			cpuset_ss << cs << ",";
		else
			cpuset_ss << cs << "-" << c << ",";
	}
	auto cpuset_str = cpuset_ss.str();
	cpuset_str.pop_back(); // Chop off trailing comma
	return cpuset_str;
}

static std::vector<cpu_set_t> get_numa_cpusets()
{
	if (numa_available() == -1) {
		fprintf(stderr,
			"WARNING: NUMA not available, not spreading threads among NUMA nodes\n");
		return std::vector<cpu_set_t>();
	}

	int numa_max_nodes = numa_max_node(), nprocs = get_nprocs();
	struct bitmask *bm = numa_get_mems_allowed();
	std::unique_ptr<struct bitmask, decltype(&numa_free_cpumask)>
		cpumask { numa_allocate_cpumask(), numa_free_cpumask };
	std::vector<cpu_set_t> numa_cpusets;

	for (int n = 0; n <= numa_max_nodes; n++) {
		if (!numa_bitmask_isbitset(bm, n))
			continue;

		numa_node_to_cpus(n, cpumask.get());

		cpu_set_t cpuset;
		CPU_ZERO(&cpuset);
		for (int c = 0; c < nprocs; c++) {
			if (!numa_bitmask_isbitset(cpumask.get(), c))
				continue;

			CPU_SET(c, &cpuset);
		}
		numa_cpusets.push_back(cpuset);
	}

	for (size_t n = 0; n < numa_cpusets.size(); n++)
		printf("NUMA CPU set %zu contains %s\n", n,
			cpu_set_to_string(&numa_cpusets[n]).c_str());

	return numa_cpusets;
}

static std::vector<cpu_set_t> numa_cpusets = get_numa_cpusets();

static void spread_threads_among_numa_nodes(std::vector<std::thread> &threads)
{
	if (numa_cpusets.empty())
		return;

	for (size_t i = 0; i < threads.size(); i++) {
		cpu_set_t cpuset = numa_cpusets[i % numa_cpusets.size()];
		int err = pthread_setaffinity_np(threads[i].native_handle(), sizeof(cpu_set_t), &cpuset);
		if (err != 0) {
			fprintf(stderr, "WARNING: Error setting thread affinity for NUMA spread (%d): %s\n\n",
				err, strerror(err));
			return;
		}
	}
}
#endif

class compdecomp_threads {
public:
	compdecomp_threads(const char *thread_kind,
			   unsigned int num_threads,
			   const std::function<int(size_t)> &worker_func)
		: thread_kind(thread_kind), worker_func(worker_func),
		  threads(num_threads), handled_chunks_per_thread(num_threads, 0) {
		for (size_t thread_idx = 0; thread_idx < threads.size(); thread_idx++) {
			threads[thread_idx] = std::thread([thread_idx, this, &worker_func] {
				// Notify the owner we're ready
				{
					std::unique_lock<std::mutex> lock(thread_mutex);
					threads_ready++;
					thread_trigger.notify_all();
				}

				while (true) {
					// Wait for work on the queue, or an exit message
					std::unique_lock<std::mutex> lock(thread_mutex);
					thread_trigger.wait(lock, [this] {
						return !thread_queue.empty() || thread_exit;
					});
					if (thread_queue.empty() && thread_exit)
						break;
					size_t chunk_num = thread_queue.front();
					thread_queue.pop();
					lock.unlock();
					handled_chunks_per_thread[thread_idx]++;

					// Do the actual work
					int err = worker_func(chunk_num);
					if (err != 0 && !thread_error.exchange(true)) {
						fprintf(stderr, "FAIL: Error during %s (%d): %s\n",
						        this->thread_kind, err, strerror(err));
					}
				}
			});
		}

#ifdef SPREAD_THREADS_AMONG_NUMA_NODES
		spread_threads_among_numa_nodes(threads);
#endif

		// Wait until the threads are actually running, as to
		// not to include thread spawning overhead in the timing
		{
			std::unique_lock<std::mutex> lock(thread_mutex);
			thread_trigger.wait(lock, [this] {
				return threads_ready == threads.size();
			});
		}
	}

	bool push_work(size_t num_chunks) {
		for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
			std::unique_lock<std::mutex> lock(thread_mutex);
			thread_queue.push(chunk_num);
			thread_trigger.notify_one();
		}
		thread_exit = true;
		thread_trigger.notify_all();
		for (size_t thread_idx = 0; thread_idx < threads.size(); thread_idx++) {
			threads[thread_idx].join();
			printf("Thread for %s %zu handled %zu chunks\n", thread_kind,
			       thread_idx, handled_chunks_per_thread[thread_idx]);
		}

		return !thread_error;
	}
private:
	unsigned int threads_ready = 0;
	bool thread_exit = false;
	std::mutex thread_mutex;
	std::condition_variable thread_trigger;
	std::queue<size_t> thread_queue;
	std::atomic<bool> thread_error{false};

	const char *thread_kind;
	std::function<int(size_t)> worker_func;
	std::vector<std::thread> threads;
	std::vector<size_t> handled_chunks_per_thread;
};

bool compress_benchmark_core(const uint8_t *in, size_t ilen,
			     uint8_t *out, size_t *olen,
			     uint8_t *decompressed, size_t *dlen,
			     long long *time_comp,
			     long long *time_condense,
			     long long *time_decomp) {
	// -----
	// SETUP
	// -----
	unsigned int num_threads = determine_num_threads();

	size_t num_chunks = ilen / CHUNK_SIZE;
	std::vector<size_t> compressed_chunk_sizes(num_chunks);
	std::vector<size_t> decompressed_chunk_sizes(num_chunks);

	// -----------
	// COMPRESSION
	// -----------
	// TODOXXX: Why is the performance on the first run so horrible?
	//          Is it due to NUMA effects?Why does it not happen with OpenMP?
	/* for (int i = 0; i < 2; i++) */ {
	std::function<int(size_t)> comp_func =
		[&in, &out, &compressed_chunk_sizes]
		(size_t chunk_num) -> int {
		size_t chunk_olen = CHUNK_SIZE * 2;
		const uint8_t *chunk_in = in + (CHUNK_SIZE * chunk_num);
		uint8_t *chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);
		int err = lib842_compress(chunk_in, CHUNK_SIZE, chunk_out,
					  &chunk_olen);
		compressed_chunk_sizes[chunk_num] = chunk_olen;
		return -err;
	};
	compdecomp_threads comp_threads("compression", num_threads, comp_func);

	long long timestart_comp = timestamp();
	if (!comp_threads.push_work(num_chunks))
		return false;
	*time_comp = timestamp() - timestart_comp;

	// printf("TIME COMP%i: %lli\n", i, *time_comp);
	}

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
	std::function<int(size_t)> decomp_func =
		[&out, &decompressed, &compressed_chunk_sizes, &decompressed_chunk_sizes]
		(size_t chunk_num) -> int {
		size_t chunk_dlen = CHUNK_SIZE;
		uint8_t *chunk_out = out + ((CHUNK_SIZE * 2) * chunk_num);
		uint8_t *chunk_decomp = decompressed + (CHUNK_SIZE * chunk_num);

		int err = lib842_decompress(chunk_out,
					    compressed_chunk_sizes[chunk_num],
					    chunk_decomp, &chunk_dlen);
		decompressed_chunk_sizes[chunk_num] = chunk_dlen;
		return -err;
	};
	compdecomp_threads decomp_threads("decompression", num_threads, decomp_func);

	long long timestart_decomp = timestamp();
	if (!decomp_threads.push_work(num_chunks))
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
	return compdecomp(argc > 1 ? argv[1] : NULL, CHUNK_SIZE, ALIGNMENT);
}
