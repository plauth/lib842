#include "cl842decompress.hpp"

size_t CL842Decompress::paddedSize(size_t size) {
    size_t workgroup_size = CHUNK_SIZE * LOCAL_SIZE;
    return (size + (workgroup_size-1)) & ~(workgroup_size-1);
}

CL842Decompress::CL842Decompress(uint8_t* input, size_t inputSize, uint8_t* output, size_t outputSize) {
    m_inputHostMemory = input;
    m_inputSize = inputSize;
    m_outputHostMemory = output;
    m_outputSize = outputSize;
    m_numChunks = inputSize / (CHUNK_SIZE * 2);

    cl::Platform::get(&m_platforms);
    if(m_platforms.empty()) {
        std::cerr << "No OpenCL platforms are available!" << std::endl;
        exit(-1);
    }
    std::cerr << "Number of available platforms: " << m_platforms.size() << std::endl;

    for(auto platform = m_platforms.begin(); m_devices.empty() && platform != m_platforms.end(); platform++) {
        std::vector<cl::Device> platformDevices;
        platform->getDevices(CL_DEVICE_TYPE_GPU, &platformDevices);
        if(platformDevices.empty()) continue;


        std::cerr << "Platform: " << platform->getInfo<CL_PLATFORM_NAME>() << std::endl;
        for(auto device = platformDevices.begin(); device != platformDevices.end(); device++) {
            if (!device->getInfo<CL_DEVICE_AVAILABLE>()) continue;
            std::cerr << "Device: " << device->getInfo<CL_DEVICE_NAME>() << std::endl;
            m_devices.push_back(*device);
        }
    }

    if(m_devices.empty()) {
        std::cerr << "No GPU devices are available!!" << std::endl;
        exit(-1);
    }

    m_context = cl::Context(m_devices);
    m_queue = cl::CommandQueue(m_context, m_devices[0]);
    
    buildProgram(kernelSource());

    m_inputBuffer = cl::Buffer(m_context, CL_MEM_READ_ONLY, m_inputSize);
    m_outputBuffer = cl::Buffer(m_context, CL_MEM_READ_WRITE, m_outputSize);

}

void CL842Decompress::buildProgram(std::string sourceCode) {
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

    std::ostringstream options;
    options<<"-D CHUNK_SIZE="<< CHUNK_SIZE;

    m_program = cl::Program(m_context, source);
    if (m_program.build(m_devices, options.str().c_str()) == CL_BUILD_PROGRAM_FAILURE) {
        std::cerr << "Build Log: " << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_devices[0], NULL) << std::endl;
}
}

std::string CL842Decompress::kernelSource() const{
    std::ifstream sourceFile("src/ocl/decompress.cl");
    return std::string(
        std::istreambuf_iterator<char>(sourceFile),
        (std::istreambuf_iterator<char>()));
}


void CL842Decompress::decompress() {
    cl_int err;

    m_queue.enqueueWriteBuffer(m_inputBuffer, CL_TRUE, 0, m_inputSize, m_inputHostMemory);
    checkErr(m_queue.finish(), "enqueueWriteBuffer()");

    cl::Kernel decompressKernel(m_program, "decompress", &err);
    checkErr(err, "cl::Kernel()");

    err = decompressKernel.setArg(0, m_inputBuffer);
    checkErr(err, "Kernel::setArg(0)");
    err = decompressKernel.setArg(1, m_outputBuffer);
    checkErr(err, "Kernel::setArg(1)");

    //size_t workgroup_size = getMaxWorkGroupSize(context);
    cl::NDRange globalSize(1);
    cl::NDRange localSize(1);

    if(m_numChunks > LOCAL_SIZE) {
        std::cerr << "Using " << m_numChunks << " chunks of " << CHUNK_SIZE << " bytes, " << LOCAL_SIZE << " threads per workgroup" << std::endl;
        globalSize = cl::NDRange(m_numChunks);
        localSize = cl::NDRange(LOCAL_SIZE);
    } 
    
    fprintf(stderr, "enqueueing kernel\n");
    err = m_queue.enqueueNDRangeKernel(decompressKernel, cl::NullRange, globalSize, localSize);

    checkErr(err, "enqueueNDRangeKernel()");
    checkErr(m_queue.finish(), "execute kernel");

    m_queue.enqueueReadBuffer(m_outputBuffer, CL_TRUE, 0, m_outputSize, m_outputHostMemory);
    checkErr(m_queue.finish(), "enqueueReadBuffer()");

    return;
}
