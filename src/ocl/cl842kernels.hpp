#ifndef CL842KERNELS_HPP
#define CL842KERNELS_HPP

#include <fstream>
#include <iostream>
#include "clutil.hpp"

#define THREADS_PER_BLOCK 1
#ifndef CHUNK_SIZE
#define CHUNK_SIZE 1024
#endif

class CL842Kernels
{
    private:
    	cl::Context context;
    	cl::CommandQueue queue;

		void prepareDecompressKernel();
		static std::once_flag decompressCompileFlag;
		static cl::Program decompressProg;
		static cl::Kernel decompressKernel;
    public:
    	CL842Kernels();
    	CL842Kernels(cl::Context context, cl::CommandQueue queue);
    	cl::Buffer allocateBuffer(size_t size, cl_mem_flags flags);
    	void writeBuffer(cl::Buffer buffer, const void * ptr, size_t size);
    	void readBuffer(cl::Buffer buffer, void * ptr, size_t size);
        void fillBuffer(cl::Buffer buffer, cl_uint value, size_t offset, size_t size);
    	void decompress(cl::Buffer in, cl::Buffer out, uint32_t num_chunks);
};

#endif // CL842KERNELS_HPP
