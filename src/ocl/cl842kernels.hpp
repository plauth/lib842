#ifndef CL842KERNELS_HPP
#define CL842KERNELS_HPP

#include <fstream>
#include <iostream>
#include "clutil.hpp"

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
    	int decompress(cl::Buffer in, uint64_t ilen, cl::Buffer out, cl::Buffer olen);
};

#endif // CL842KERNELS_HPP
