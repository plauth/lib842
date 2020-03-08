#ifndef __CL842_H__
#define __CL842_H__

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <stdint.h>

#ifndef CL842_CHUNK_SIZE
#define CL842_CHUNK_SIZE 65536
#endif

static const uint8_t CL842_COMPRESSED_CHUNK_MAGIC[16] = {
        0xbe, 0x5a, 0x46, 0xbf, 0x97, 0xe5, 0x2d, 0xd7, 0xb2, 0x7c, 0x94, 0x1a, 0xee, 0xd6, 0x70, 0x76
};

/**
 * Low-level interface to CL842, for integration into existing OpenCL applications
 * where context, command queue, buffers, etc. are already available.
 */
class CL842DeviceDecompressor
{
    public:
        CL842DeviceDecompressor(const cl::Context& context,
                                const VECTOR_CLASS<cl::Device>& devices,
                                size_t inputChunkStride,
                                bool verbose = false,
                                bool inPlace = false);
        void decompress(const cl::CommandQueue& commandQueue,
                        const cl::Buffer& inputBuffer, size_t inputOffset, size_t inputSize,
                        const cl::Buffer& outputBuffer, size_t outputOffset, size_t outputSize,
                        const VECTOR_CLASS<cl::Event>* events = nullptr, cl::Event* event = nullptr);

    private:
        bool m_verbose, m_inPlace;
        cl::Program m_program;

        size_t m_inputChunkStride;

        void buildProgram(const cl::Context& context, const VECTOR_CLASS<cl::Device>& devices);
};

/**
 * High-level interface to CL842, for easily compressing data available on the host
 * using any available OpenCL-capable devices.
 */
class CL842HostDecompressor
{
    public:
        static size_t paddedSize(size_t size);

        CL842HostDecompressor(size_t inputChunkStride,
                              bool verbose = false,
                              bool inPlace = false);
        void decompress(const uint8_t* input, size_t inputSize,
                        uint8_t* output, size_t outputSize);

    private:
        bool m_verbose, m_inPlace;
        VECTOR_CLASS<cl::Device> m_devices;
        cl::Context m_context;
        cl::CommandQueue m_queue;
        CL842DeviceDecompressor deviceCompressor;

        VECTOR_CLASS<cl::Device> findDevices();
};

#endif
