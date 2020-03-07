#include "../../include/cl842.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <chrono>

// This symbol is generated during the build process and defines a
// CL842_DECOMPRESS_SOURCE with the source of the decompression OpenCL kernel
extern const char *CL842_DECOMPRESS_SOURCE;

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 256
#endif

CL842DeviceDecompressor::CL842DeviceDecompressor(const cl::Context& context,
                                                 const VECTOR_CLASS<cl::Device>& devices,
                                                 size_t inputChunkStride,
                                                 bool verbose,
                                                 bool inPlace)
    : m_verbose(verbose), m_inPlace(inPlace),
      m_inputChunkStride(inputChunkStride)
{
    buildProgram(context, devices);
}


void CL842DeviceDecompressor::decompress(const cl::CommandQueue& commandQueue,
                                         const cl::Buffer& inputBuffer, size_t inputSize,
                                         const cl::Buffer& outputBuffer, size_t outputSize,
                                         const VECTOR_CLASS<cl::Event>* events, cl::Event* event)
{
    if (m_inPlace) {
        if (inputBuffer() != outputBuffer() || inputSize != outputSize) {
            throw cl::Error(CL_INVALID_VALUE);
        }
    }
    size_t numChunks = (inputSize + m_inputChunkStride - 1) / (m_inputChunkStride); 

    cl::Kernel decompressKernel(m_program, "decompress");

    decompressKernel.setArg(0, inputBuffer);
    decompressKernel.setArg(1, outputBuffer);
    decompressKernel.setArg(2, static_cast<cl_ulong>(numChunks));

    cl::NDRange globalSize((numChunks + (LOCAL_SIZE-1)) & ~(LOCAL_SIZE-1));
    cl::NDRange localSize(LOCAL_SIZE);

    if(numChunks > 1 && m_verbose) {
        std::cerr << "Using " << numChunks << " chunks of " << CL842_CHUNK_SIZE << " bytes, " << LOCAL_SIZE << " threads per workgroup" << std::endl;
    }

    std::chrono::steady_clock::time_point t1;
    if (m_verbose) {
        t1 = std::chrono::steady_clock::now();
    }

    commandQueue.enqueueNDRangeKernel(decompressKernel, cl::NullRange, globalSize, localSize, events, event);

    if (m_verbose) {
        commandQueue.finish();

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

        std::cerr << "Decompression performance: "
            << duration << "ms" << " / "
            << (outputSize / 1024 / 1024) / ((float) duration / 1000) << "MiB/s"
            << std::endl;
    }
}

void CL842DeviceDecompressor::buildProgram(const cl::Context& context, const VECTOR_CLASS<cl::Device>& devices) {
    std::ostringstream options;
    options<<"-D CL842_CHUNK_SIZE="<< CL842_CHUNK_SIZE;
    options<<" -D CL842_CHUNK_STRIDE=" << m_inputChunkStride;
    if (m_inPlace)
        options<<" -D INPLACE=1";

    m_program = cl::Program(context, std::string(CL842_DECOMPRESS_SOURCE));
    try {
        m_program.build(devices, options.str().c_str());
    } catch (const cl::Error& ex) {
        if (ex.err() == CL_BUILD_PROGRAM_FAILURE && m_verbose) {
            std::cerr << "Build Log: " << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0], NULL) << std::endl;
        }
        throw;
    }
}

size_t CL842HostDecompressor::paddedSize(size_t size) {
    return (size + (CL842_CHUNK_SIZE-1)) & ~(CL842_CHUNK_SIZE-1);
}

CL842HostDecompressor::CL842HostDecompressor(size_t inputChunkStride, bool verbose, bool inPlace) :
    m_verbose(verbose), m_inPlace(inPlace),
    m_devices(findDevices()),
    m_context(m_devices),
    m_queue(m_context, m_devices[0]),
    deviceCompressor(m_context, m_devices, inputChunkStride, verbose, inPlace)
{ }

VECTOR_CLASS<cl::Device> CL842HostDecompressor::findDevices() {
    VECTOR_CLASS<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) {
        if (m_verbose) {
            std::cerr << "No OpenCL platforms are available!" << std::endl;
        }
        throw cl::Error(CL_DEVICE_NOT_FOUND);
    }
    if (m_verbose) {
        std::cerr << "Number of available platforms: " << platforms.size() << std::endl;
    }

    VECTOR_CLASS<cl::Device> devices;

    for(auto platform = platforms.begin(); devices.empty() && platform != platforms.end(); platform++) {
        VECTOR_CLASS<cl::Device> platformDevices;
        platform->getDevices(CL_DEVICE_TYPE_GPU, &platformDevices);
        if(platformDevices.empty())
            continue;

        if (m_verbose) {
            std::cerr << "Platform: " << platform->getInfo<CL_PLATFORM_NAME>() << std::endl;
        }
        for(auto device = platformDevices.begin(); device != platformDevices.end(); device++) {
            if (!device->getInfo<CL_DEVICE_AVAILABLE>())
                continue;
            if (m_verbose) {
                std::cerr << "Device: " << device->getInfo<CL_DEVICE_NAME>() << std::endl;
            }
            devices.push_back(*device);
        }
    }

    if(devices.empty()) {
        if (m_verbose) {
            std::cerr << "No GPU devices are available!!" << std::endl;
        }
        throw cl::Error(CL_DEVICE_NOT_FOUND);
    }

    return devices;
}


void CL842HostDecompressor::decompress(const uint8_t* input, size_t inputSize, uint8_t* output, size_t outputSize) {
    if (m_inPlace) {
        if (input != output || inputSize != outputSize) {
            throw cl::Error(CL_INVALID_VALUE);
        }

        // Add some more bytes for potential excess lookahead
        cl::Buffer buffer(m_context, CL_MEM_READ_WRITE, inputSize + 64);

        m_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, inputSize, input);
        deviceCompressor.decompress(m_queue, buffer, inputSize, buffer, outputSize);
        m_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, outputSize, output);
    } else {
        cl::Buffer inputBuffer(m_context, CL_MEM_READ_ONLY, inputSize);
        cl::Buffer outputBuffer(m_context, CL_MEM_READ_WRITE, outputSize);

        m_queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, input);
        deviceCompressor.decompress(m_queue, inputBuffer, inputSize, outputBuffer, outputSize);
        m_queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputSize, output);
    }
}
