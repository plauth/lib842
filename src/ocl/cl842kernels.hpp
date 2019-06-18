#ifndef CL842KERNELS_HPP
#define CL842KERNELS_HPP

#include <fstream>
#include <iostream>
#include <sstream>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#define THREADS_PER_BLOCK 256
#define CHUNK_SIZE 1024


inline void checkErr(cl_int err, const char * name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name  << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

class CL842Kernels
{
    public:
        static const cl_device_type usedDeviceTypes;
        CL842Kernels();
        cl::Buffer allocateBuffer(size_t size, cl_mem_flags flags);
        void writeBuffer(cl::Buffer buffer, const void * ptr, size_t size);
        void readBuffer(cl::Buffer buffer, void * ptr, size_t size);
        void fillBuffer(cl::Buffer buffer, cl_uint value, size_t offset, size_t size);
        void decompress(cl::Buffer in, cl::Buffer out, uint32_t num_chunks);
    private:
        //cl::Buffer compressedD, decompressedD;

        std::vector<cl::Platform> m_platforms;
        cl::Context m_context;
        std::vector<cl::Device> m_devices;
        cl::Program m_program;
        cl::CommandQueue m_queue;

        std::string decompressKernelSource() const;

};

#endif // CL842KERNELS_HPP
