#ifndef CL842DecompressKERNELS_HPP
#define CL842DecompressKERNELS_HPP

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

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 256
#endif
#ifndef CHUNK_SIZE
#define CHUNK_SIZE 1024
#endif


inline void checkErr(cl_int err, const char * name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name  << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

class CL842Decompress
{
    public:
        static size_t paddedSize(size_t size);

        CL842Decompress(uint8_t* input, size_t inputSize, uint8_t* output, size_t outputSize);
        void decompress();

    private:
        std::vector<cl::Platform> m_platforms;
        cl::Context m_context;
        std::vector<cl::Device> m_devices;
        cl::Program m_program;
        cl::CommandQueue m_queue;

        size_t m_inputSize, m_outputSize, m_numChunks;

        uint8_t* m_inputHostMemory;
        uint8_t* m_outputHostMemory;
        cl::Buffer m_inputBuffer;
        cl::Buffer m_outputBuffer;

        std::string kernelSource() const;
        void buildProgram(std::string);

};

#endif // CL842DecompressKERNELS_HPP
