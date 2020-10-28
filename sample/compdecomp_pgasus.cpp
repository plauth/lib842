#include <sys/sysinfo.h>
#include <sys/time.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>

#include <PGASUS/base/node.hpp>
#include <PGASUS/msource/msource_types.hpp>
#include <PGASUS/msource/msource.hpp>
#include <PGASUS/tasking/tasking.hpp>

#include <lib842/sw.h>
#define lib842impl (*get_optsw842_implementation())


#define CHUNK_SIZE ((size_t)65536)

typedef std::vector<uint8_t*> Memories;


static inline uint64_t next_pow2(uint64_t x) 
{
	return x == 1 ? 1 : 1<<(64-__builtin_clzl(x-1));
}

static inline size_t getPaddedInputBufferLength(size_t input) 
{
    //assumption: all numa nodes with CPUs have the same amount of cores/threads
    size_t chunk_size_times_node_count = CHUNK_SIZE * ((size_t) get_nprocs());
    return ((input + chunk_size_times_node_count - 1) / chunk_size_times_node_count) * chunk_size_times_node_count;
}

long long timestamp()
{
	struct timeval te;
	gettimeofday(&te, NULL);
	long long ms = te.tv_sec * 1000LL + te.tv_usec / 1000;
	return ms;
}


int main(int argc, const char *argv[])
{
    int ret = EXIT_FAILURE;
    int file_length = 0;
    std::ifstream file (argc > 1 ? argv[1] : NULL, std::ifstream::binary);
    if(file)
    {
        file.seekg(0, file.end);
        file_length = file.tellg();
        file.seekg(0, file.beg);
    } else {
        std::cerr << "FAIL: Could not open the file at path '" << argv[1] << "'." << std::endl;
        return ret;
    }

    size_t total_input_buffer_length = getPaddedInputBufferLength(file_length);

    std::cout << "Opened file at path '" << argv[1] << "'." << std::endl;
    std::cout << "Original file length: " << file_length << " bytes." << std::endl;
    std::cout << "Original file length (padded): " << total_input_buffer_length << " bytes." << std::endl;
    std::cout << "Using chunks of " << CHUNK_SIZE << " bytes (" << total_input_buffer_length / CHUNK_SIZE << " chunks)." << std::endl;

    
    size_t node_input_buffer_length = total_input_buffer_length / numa::NodeList::logicalNodesWithCPUsCount();
    size_t node_compressed_buffer_length = node_input_buffer_length * 2;
    size_t node_decompressed_buffer_length = node_input_buffer_length;
    size_t node_msource_size = next_pow2(node_input_buffer_length + node_compressed_buffer_length + node_decompressed_buffer_length);


    numa::SpinLock lock;
    numa::msvector<numa::MemSource> msources(numa::MemSource::global());
    msources.resize(numa::NodeList::logicalNodesWithCPUsCount());

    for (numa::Node node : numa::NodeList::logicalNodesWithCPUs()) 
    {
        std::lock_guard<numa::SpinLock> guard(lock);
        char buff[4096];
        snprintf(buff, sizeof(buff) / sizeof(buff[0]), "workspace(node=%d, size=%zu)", node.logicalId(), node_msource_size);
        msources[node.logicalId()] = numa::MemSource::create(node, node_msource_size ,buff);
    }

    std::vector<uint8_t*> input_buffers;
    std::vector<uint8_t*> compressed_buffers;
    std::vector<uint8_t*> decompressed_buffers;
    std::vector<size_t*>  compressed_chunk_sizes;
    std::vector<size_t*>  decompressed_chunk_sizes;
    input_buffers.resize(numa::NodeList::logicalNodesWithCPUsCount());
    compressed_buffers.resize(numa::NodeList::logicalNodesWithCPUsCount());
    decompressed_buffers.resize(numa::NodeList::logicalNodesWithCPUsCount());
    compressed_chunk_sizes.resize(numa::NodeList::logicalNodesWithCPUsCount());
    decompressed_chunk_sizes.resize(numa::NodeList::logicalNodesWithCPUsCount());

    std::atomic<size_t> compressThreadIds[numa::NodeList::logicalNodesWithCPUsCount()] = {};
    std::atomic<size_t> decompressThreadIds[numa::NodeList::logicalNodesWithCPUsCount()] = {};

    size_t cpus_per_node = ((size_t) get_nprocs()) /  numa::NodeList::logicalNodesWithCPUsCount();
    size_t chunks_per_node = node_input_buffer_length / CHUNK_SIZE;
    size_t chunks_per_cpu = chunks_per_node / cpus_per_node;

    
    std::atomic_size_t threads_init(0);

	// startup worker threads
	numa::wait(numa::forEachThread(numa::NodeList::logicalNodesWithCPUs(), [&](){
        numa::Node currentNode = numa::Node::curr();
        size_t localThreadId = threads_init.fetch_add(1) % cpus_per_node;
        //printf("[Init] Hello from node %d, thread %2zu\n", currentNode.logicalId(), localThreadId);
	}, 0));
    //std::cout << "Initialized " << threads_init << " worker threads" << std::endl;

    for (numa::Node node : numa::NodeList::logicalNodesWithCPUs()) 
    {
        numa::PlaceGuard guard(msources[node.logicalId()]);

        input_buffers[node.logicalId()] = (uint8_t*) std::aligned_alloc(sizeof(uint64_t), node_input_buffer_length);
        if (input_buffers[node.logicalId()] == NULL) {
		    std::cerr << "FAIL: Could not allocate input buffer on node " << node.physicalId() << "." << std::endl;
		    return ret;
	    }
        std::memset(input_buffers[node.logicalId()], 0, node_input_buffer_length);

        compressed_buffers[node.logicalId()] = (uint8_t*) std::aligned_alloc(sizeof(uint64_t), node_compressed_buffer_length);
        if (compressed_buffers[node.logicalId()] == NULL) {
		    std::cerr << "FAIL: Could not allocate compressed buffer on node " << node.physicalId() << "." << std::endl;
		    return ret;
	    }
        std::memset(compressed_buffers[node.logicalId()], 0, node_compressed_buffer_length);

        decompressed_buffers[node.logicalId()] = (uint8_t*) std::aligned_alloc(sizeof(uint64_t), node_decompressed_buffer_length);
        if (decompressed_buffers[node.logicalId()] == NULL) {
		    std::cerr << "FAIL: Could not allocate decompressed buffer on node " << node.physicalId() << "." << std::endl;
		    return ret;
	    }
        std::memset(decompressed_buffers[node.logicalId()], 0, node_decompressed_buffer_length);

        compressed_chunk_sizes[node.logicalId()] = (size_t*) std::malloc(chunks_per_node * sizeof(size_t));
        if (compressed_chunk_sizes[node.logicalId()] == NULL) {
		    std::cerr << "FAIL: Could not allocate buffer for compressed chunk sizes on node " << node.physicalId() << "." << std::endl;
		    return ret;
	    }  

        decompressed_chunk_sizes[node.logicalId()] = (size_t*) std::malloc(chunks_per_node * sizeof(size_t));
        if (decompressed_chunk_sizes[node.logicalId()] == NULL) {
		    std::cerr << "FAIL: Could not allocate buffer for decompressed chunk sizes on node " << node.physicalId() << "." << std::endl;
		    return ret;
	    }

        compressThreadIds[node.logicalId()] = 0;
        decompressThreadIds[node.logicalId()] = 0;

        file.read((char*) input_buffers[node.logicalId()], node_input_buffer_length);
        //std::cout << "Reading " << file.gcount() << " on node " << node.logicalId() << "." << std::endl;
        //printf("Hello from Node %d/%d (phsical/logical)!\n", node.physicalId(), node.logicalId());
    }

    std::cout << "CPUs per node: " << cpus_per_node << std::endl;
    std::cout << "Chunks per Node: " << chunks_per_node << std::endl;
    std::cout << "Chunks per CPU: " << chunks_per_cpu << std::endl;

    long long timestart_comp = timestamp();

    numa::wait(numa::forEachThread(numa::NodeList::logicalNodesWithCPUs(), [&](){
        numa::Node currentNode = numa::Node::curr();
        size_t localThreadId = compressThreadIds[currentNode.logicalId()].fetch_add(1);
        //printf("[Compress] Hello from node %d, thread %2zu\n", currentNode.logicalId(), localThreadId);

        size_t chunkStartLocal = localThreadId * chunks_per_cpu;
        size_t chunkEndLocal = chunkStartLocal + chunks_per_cpu - 1;

        for(size_t chunkNum = chunkStartLocal; chunkNum <= chunkEndLocal; chunkNum++) {
            const uint8_t *chunk_in = input_buffers[currentNode.logicalId()] + (chunkNum * CHUNK_SIZE);
		    uint8_t *chunk_out = compressed_buffers[currentNode.logicalId()] + (chunkNum * (CHUNK_SIZE * 2));
            size_t *compressed_chunk_size = compressed_chunk_sizes[currentNode.logicalId()] + chunkNum;
            *compressed_chunk_size = CHUNK_SIZE * 2;
            int err = lib842impl.compress(chunk_in, CHUNK_SIZE, chunk_out, compressed_chunk_size);
            //printf("[Compress] Node %d, Thread %2zu, chunkNum %4zu\n", currentNode.logicalId(), localThreadId, chunkNum);
            if (err != 0) {
                std::lock_guard<numa::SpinLock> guard(lock);
			    std::cerr << "Error during compression: " << err <<  std::endl;
                return ret;
		    }
        }
	}, 0));
    long long timeend_comp  = timestamp() - timestart_comp;

    /*
    for (numa::Node node : numa::NodeList::logicalNodesWithCPUs()) 
    {
        std::cout << "Node " << node.logicalId() << " used " << compressThreadIds[node.logicalId()] << " worker threads for compression." << std::endl;
    }
    */

    long long timestart_decomp = timestamp();
    numa::wait(numa::forEachThread(numa::NodeList::logicalNodesWithCPUs(), [&](){
        numa::Node currentNode = numa::Node::curr();
        size_t localThreadId = decompressThreadIds[currentNode.logicalId()].fetch_add(1);
        //printf("[Decompress] Hello from node %d, thread %2zu\n", currentNode.logicalId(), localThreadId);

        size_t chunkStartLocal = localThreadId * chunks_per_cpu;
        size_t chunkEndLocal = chunkStartLocal + chunks_per_cpu - 1;

        for(size_t chunkNum = chunkStartLocal; chunkNum <= chunkEndLocal; chunkNum++) {
            const uint8_t *chunk_out = compressed_buffers[currentNode.logicalId()] + (chunkNum * (CHUNK_SIZE * 2));
		    uint8_t *chunk_decomp = decompressed_buffers[currentNode.logicalId()] + (chunkNum * CHUNK_SIZE);
            size_t *compressed_chunk_size = compressed_chunk_sizes[currentNode.logicalId()] + chunkNum;
            size_t *decompressed_chunk_size = decompressed_chunk_sizes[currentNode.logicalId()] + chunkNum;
            *decompressed_chunk_size = CHUNK_SIZE;

            int err = lib842impl.decompress(chunk_out, *compressed_chunk_size, chunk_decomp, decompressed_chunk_size);
            if (err != 0) {
                std::lock_guard<numa::SpinLock> guard(lock);
			    std::cerr << "Error during decompression: " << err <<  std::endl;
                return ret;
		    }
        }
        //printf("Decompression: Hello from NUMA node %d, CPU %3d / %3d (local / global)! This thread processed %zu chunks (from chunk %5zu through %5zu.\n", currentNode.logicalId(), currentNode.indexOfCpuid(currentNode.currCpuid()), currentNode.currCpuid(), chunksProcessed, chunkStartGlobal, chunkEndGlobal);
	}, 0));    
    long long timeend_decomp  = timestamp() - timestart_decomp;

    /*
    for (numa::Node node : numa::NodeList::logicalNodesWithCPUs()) 
    {
        std::cout << "Node " << node.logicalId() << " used " << decompressThreadIds[node.logicalId()] << " worker threads for decompression." << std::endl;
    }
    */   

    for (numa::Node node : numa::NodeList::logicalNodesWithCPUs()) 
    {
        if (memcmp(input_buffers[node.logicalId()], decompressed_buffers[node.logicalId()], node_input_buffer_length) != 0)
            std::cerr << "FAIL: Decompressed data differs from the original input data on node " << node.logicalId() << "." << std::endl;
    }





    /*
    // query node location of memory address
    int status = -1;
    int ret_code;
    ret_code = move_pages(0, 1, &tmp_input_buffer, NULL, &status, 0);
    printf("Memory at %p is at %d node (retcode %d)\n", tmp_input_buffer, status, ret_code);
    */
    printf("Compression performance: %lld ms / %f MiB/s\n",
            timeend_comp, (total_input_buffer_length / 1024 / 1024) / ((float)timeend_comp / 1000));
    printf("Decompression performance: %lld ms / %f MiB/s\n",
        timeend_decomp, (total_input_buffer_length / 1024 / 1024) / ((float)timeend_decomp / 1000));
}


