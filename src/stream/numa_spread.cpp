#include "numa_spread.h"

#include <lib842/stream/common.h>
#include <cstdio>

#ifdef LIB842_HAVE_NUMA

#include <pthread.h>
#include <numa.h>
#include <sys/sysinfo.h>
#ifdef LIB842_STREAM_INDEPTH_TRACE
#include <string>
#include <sstream>
#endif

#ifdef LIB842_STREAM_INDEPTH_TRACE
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
#endif

static std::vector<cpu_set_t> get_numa_cpusets()
{
	if (numa_available() == -1) {
		fprintf(stderr,
			"WARNING: NUMA not available, not spreading threads among NUMA nodes\n");
		return std::vector<cpu_set_t>();
	}

	int numa_max_nodes = numa_max_node(), nprocs = get_nprocs();
	std::unique_ptr<struct bitmask, decltype(&numa_free_cpumask)>
		bm { numa_get_mems_allowed(), numa_free_cpumask };
	std::unique_ptr<struct bitmask, decltype(&numa_free_cpumask)>
		cpumask { numa_allocate_cpumask(), numa_free_cpumask };
	std::vector<cpu_set_t> numa_cpusets;

	for (int n = 0; n <= numa_max_nodes; n++) {
		if (!numa_bitmask_isbitset(bm.get(), n))
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

#ifdef LIB842_STREAM_INDEPTH_TRACE
	for (size_t n = 0; n < numa_cpusets.size(); n++)
		printf("NUMA CPU set %zu contains %s\n", n,
			cpu_set_to_string(&numa_cpusets[n]).c_str());
#endif

	return numa_cpusets;
}

#endif

void spread_threads_among_numa_nodes(std::vector<std::thread> &threads)
{
#ifndef LIB842_HAVE_NUMA
	fprintf(stderr,
		"WARNING: libNUMA or pthreads not available, not spreading threads among NUMA nodes\n");
#else
	static std::vector<cpu_set_t> numa_cpusets = get_numa_cpusets();
	if (numa_cpusets.empty())
		return;

	for (size_t i = 0; i < threads.size(); i++) {
		cpu_set_t cpuset = numa_cpusets[i % numa_cpusets.size()];
#ifdef LIB842_STREAM_INDEPTH_TRACE
		printf("Assigning thread %zu to CPU set %s\n",
			i, cpu_set_to_string(&cpuset).c_str());
#endif
		int err = pthread_setaffinity_np(threads[i].native_handle(), sizeof(cpu_set_t), &cpuset);
		if (err != 0) {
			fprintf(stderr, "WARNING: Error setting thread affinity for NUMA spread (%d): %s\n\n",
				err, strerror(err));
			return;
		}
	}
#endif
}


