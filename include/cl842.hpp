#ifndef __CL842_H__
#define __CL842_H__

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <stdint.h>

#ifndef CL842_CHUNK_SIZE
#define CL842_CHUNK_SIZE 1024
#endif

class CL842DeviceDecompress
{
    public:
        CL842DeviceDecompress(const cl::Context &context, const VECTOR_CLASS<cl::Device> &devices, size_t inputChunkStride);
        void decompress(const cl::CommandQueue &commandQueue,
                        const cl::Buffer &inputBuffer, size_t inputSize,
                        const cl::Buffer &outputBuffer, size_t outputSize,
                        const VECTOR_CLASS<cl::Event> *events = nullptr, cl::Event *event = nullptr);

    private:
        cl::Program m_program;

        size_t m_inputChunkStride;

        std::string kernelSource() const;
        void buildProgram(const cl::Context &context, const VECTOR_CLASS<cl::Device> &devices, std::string sourceCode);
};

class CL842HostDecompress
{
    public:
        static size_t paddedSize(size_t size);

        CL842HostDecompress(size_t inputChunkStride);
        void decompress(const uint8_t* input, size_t inputSize, uint8_t* output, size_t outputSize);

    private:
        std::vector<cl::Device> m_devices;
        cl::Context m_context;
        cl::CommandQueue m_queue;
        CL842DeviceDecompress backend;

        static std::vector<cl::Device> findDevices();
};

#endif
