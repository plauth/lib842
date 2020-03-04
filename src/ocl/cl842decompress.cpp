#include <chrono>
#include "../../include/cl842.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 256
#endif

static inline void checkErr(cl_int err, const char * name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name  << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

CL842DeviceDecompress::CL842DeviceDecompress(const cl::Context &context, const VECTOR_CLASS<cl::Device> &devices, size_t inputChunkStride)
    : m_inputChunkStride(inputChunkStride)
{
    buildProgram(context, devices, kernelSource());
}


void CL842DeviceDecompress::decompress(const cl::CommandQueue &commandQueue,
                                         const cl::Buffer &inputBuffer, size_t inputSize,
                                         const cl::Buffer &outputBuffer, size_t outputSize,
                                         const VECTOR_CLASS<cl::Event> *events, cl::Event *event)
{
    cl_int err;

    size_t numChunks = (inputSize + m_inputChunkStride - 1) / (m_inputChunkStride); 

    cl::Kernel decompressKernel(m_program, "decompress", &err);
    checkErr(err, "cl::Kernel()");

    err = decompressKernel.setArg(0, inputBuffer);
    checkErr(err, "Kernel::setArg(0)");
    err = decompressKernel.setArg(1, outputBuffer);
    checkErr(err, "Kernel::setArg(1)");
    err = decompressKernel.setArg(2, static_cast<cl_ulong>(numChunks));
    checkErr(err, "Kernel::setArg(2)");

    cl::NDRange globalSize((numChunks + (LOCAL_SIZE-1)) & ~(LOCAL_SIZE-1));
    cl::NDRange localSize(LOCAL_SIZE);

    if(numChunks > 1) {
        std::cerr << "Using " << numChunks << " chunks of " << CL842_CHUNK_SIZE << " bytes, " << LOCAL_SIZE << " threads per workgroup" << std::endl;
    }

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    err = commandQueue.enqueueNDRangeKernel(decompressKernel, cl::NullRange, globalSize, localSize, events, event);

    checkErr(err, "enqueueNDRangeKernel()");
    checkErr(commandQueue.finish(), "execute kernel");

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

    std::cerr << "Decompression performance: " << duration << "ms" << " / " << (outputSize / 1024 / 1024) / ((float) duration / 1000) << "MiB/s" << std::endl;
}

std::string CL842DeviceDecompress::kernelSource() const{
    std::ifstream sourceFile("src/ocl/decompress.cl");
    return std::string(
        std::istreambuf_iterator<char>(sourceFile),
        (std::istreambuf_iterator<char>()));
}

void CL842DeviceDecompress::buildProgram(const cl::Context &context, const VECTOR_CLASS<cl::Device> &devices, std::string sourceCode) {
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

    std::ostringstream options;
    options<<"-D CL842_CHUNK_SIZE="<< CL842_CHUNK_SIZE;
    options<<" -D CL842_CHUNK_STRIDE=" << m_inputChunkStride;

    m_program = cl::Program(context, source);
    if (m_program.build(devices, options.str().c_str()) == CL_BUILD_PROGRAM_FAILURE) {
        std::cerr << "Build Log: " << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0], NULL) << std::endl;
    }
}

size_t CL842HostDecompress::paddedSize(size_t size) {
    return (size + (CL842_CHUNK_SIZE-1)) & ~(CL842_CHUNK_SIZE-1);
}

CL842HostDecompress::CL842HostDecompress(size_t inputChunkStride) :
    m_devices(findDevices()),
    m_context(m_devices),
    m_queue(m_context, m_devices[0]),
    backend(m_context, m_devices, inputChunkStride)
{ }

std::vector<cl::Device> CL842HostDecompress::findDevices() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) {
        std::cerr << "No OpenCL platforms are available!" << std::endl;
        exit(-1);
    }
    std::cerr << "Number of available platforms: " << platforms.size() << std::endl;

    std::vector<cl::Device> devices;

    for(auto platform = platforms.begin(); devices.empty() && platform != platforms.end(); platform++) {
        std::vector<cl::Device> platformDevices;
        platform->getDevices(CL_DEVICE_TYPE_GPU, &platformDevices);
        if(platformDevices.empty()) continue;


        std::cerr << "Platform: " << platform->getInfo<CL_PLATFORM_NAME>() << std::endl;
        for(auto device = platformDevices.begin(); device != platformDevices.end(); device++) {
            if (!device->getInfo<CL_DEVICE_AVAILABLE>()) continue;
            std::cerr << "Device: " << device->getInfo<CL_DEVICE_NAME>() << std::endl;
            devices.push_back(*device);
        }
    }

    if(devices.empty()) {
        std::cerr << "No GPU devices are available!!" << std::endl;
        exit(-1);
    }

    return devices;
}


void CL842HostDecompress::decompress(const uint8_t* input, size_t inputSize, uint8_t* output, size_t outputSize) {

    cl::Buffer inputBuffer(m_context, CL_MEM_READ_ONLY, inputSize);
    cl::Buffer outputBuffer(m_context, CL_MEM_READ_WRITE, outputSize);

    m_queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, input);
    checkErr(m_queue.finish(), "enqueueWriteBuffer()");

    backend.decompress(m_queue, inputBuffer, inputSize, outputBuffer, outputSize);

    m_queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputSize, output);
    checkErr(m_queue.finish(), "enqueueReadBuffer()");

    return;
}
