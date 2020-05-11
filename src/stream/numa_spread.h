#ifndef LIB842_SRC_STREAM_NUMA_SPREAD_H
#define LIB842_SRC_STREAM_NUMA_SPREAD_H

#include <vector>
#include <thread>

// When called, threads will be distributed evenly among available NUMA nodes
// This should provide an effect like OpenMP's OMP_PLACES=sockets OMP_PROC_BIND=spread
void spread_threads_among_numa_nodes(std::vector<std::thread> &threads);

#endif
