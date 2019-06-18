#include "cl842kernels.hpp"

const cl_device_type CL842Kernels::usedDeviceTypes = CL_DEVICE_TYPE_ALL;

CL842Kernels::CL842Kernels() {

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

        std::string sourceCode = decompressKernelSource();
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

        std::ostringstream options;
        options<<"-D CHUNK_SIZE="<< CHUNK_SIZE;

        m_program = cl::Program(m_context, source);
        if (m_program.build(m_devices, options.str().c_str()) == CL_BUILD_PROGRAM_FAILURE) {
            std::cerr << "Build Log: " << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_devices[0], NULL) << std::endl;
        }
        
        m_queue = cl::CommandQueue(m_context, m_devices[0]);
}

std::string CL842Kernels::decompressKernelSource() const{
    std::ifstream sourceFile("src/ocl/decompress.cl");
    return std::string(
        std::istreambuf_iterator<char>(sourceFile),
        (std::istreambuf_iterator<char>()));
}


void CL842Kernels::decompress(cl::Buffer in, cl::Buffer out, uint32_t num_chunks) {
    cl_int err;

    cl::Kernel decompressKernel(m_program, "decompress", &err);
    checkErr(err, "cl::Kernel()");

    err = decompressKernel.setArg(0, in);
    checkErr(err, "Kernel::setArg(0)");
    err = decompressKernel.setArg(1, out);
    checkErr(err, "Kernel::setArg(1)");

    //size_t workgroup_size = getMaxWorkGroupSize(context);
    cl::NDRange globalSize(1);
    cl::NDRange localSize(1);

    if(num_chunks > LOCAL_SIZE) {
        fprintf(stderr, "Using %d chunks of %d bytes, %d threads per workgroup\n", num_chunks, CHUNK_SIZE, LOCAL_SIZE);
        globalSize = cl::NDRange(num_chunks);
        localSize = cl::NDRange(LOCAL_SIZE);
    } 
    
    fprintf(stderr, "enqueueing kernel\n");
    err = m_queue.enqueueNDRangeKernel(decompressKernel, cl::NullRange, globalSize, localSize);
    checkErr(err, "enqueueNDRangeKernel()");
    checkErr(m_queue.finish(), "execute kernel");
    return;
}

cl::Buffer CL842Kernels::allocateBuffer(size_t size, cl_mem_flags flags) {
    cl_int err;
    cl::Buffer buffer = cl::Buffer(m_context, flags, size, NULL, &err);
    checkErr(err, "cl::Buffer()");
    return buffer;
}

void CL842Kernels::writeBuffer(cl::Buffer buffer, const void * ptr, size_t size) {
    m_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size, ptr);
    checkErr(m_queue.finish(), "enqueueWriteBuffer()");
}

void CL842Kernels::readBuffer(cl::Buffer buffer, void * ptr, size_t size) {
    m_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, ptr);
    checkErr(m_queue.finish(), "enqueueReadBuffer()");
}

void CL842Kernels::fillBuffer(cl::Buffer buffer, cl_uint value, size_t offset, size_t size) {
    m_queue.enqueueFillBuffer(buffer, value, offset, size);
    checkErr(m_queue.finish(), "enqueueFillBuffer()");
}

size_t CL842Kernels::paddedSize(size_t size) {
    size_t workgroup_size = CHUNK_SIZE * LOCAL_SIZE;
    return (size + (workgroup_size-1)) & ~(workgroup_size-1);
}

