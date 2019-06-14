#include "cl842kernels.hpp"

using namespace std;

CL842Kernels::CL842Kernels() {
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> devices;

        cl::Platform::get(&platforms);
        if(platforms.empty()) {
            std::cerr << "No OpenCL platforms are available!" << std::endl;
            exit(-1);
        }
        std::cerr << "Number of available platforms: " << platforms.size() << std::endl;

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
        context = cl::Context(devices);

        queue = cl::CommandQueue(context, devices[0]);
}

void CL842Kernels::prepareDecompressKernel() {
    std::cerr << "Compiling decompress kernel..." << std::endl;
    
    cl_int err;
    ifstream cl_file("src/ocl/decompress.cl");
    string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
   
    std::ostringstream options;
    options<<"-D CHUNK_SIZE="<< CHUNK_SIZE;

    decompressProg = cl::Program (context, cl_string.c_str());
    err = decompressProg.build(options.str().c_str());

    if (err == CL_BUILD_PROGRAM_FAILURE) {
        cl::vector<cl::Device> devices;
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");
        std::string log = decompressProg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0], NULL);
        std::cerr << "Build Log: " << log << std::endl;
    }
    checkErr(err, "cl::Programm::build()");

    decompressKernel = cl::Kernel(decompressProg, "decompress", &err);
    checkErr(err, "cl::Kernel()");
        
}

void CL842Kernels::decompress(cl::Buffer in, cl::Buffer out, uint32_t num_chunks) {
    cl_int err;

    err = decompressKernel.setArg(0, in);
    checkErr(err, "Kernel::setArg(0)");
    err = decompressKernel.setArg(1, out);
    checkErr(err, "Kernel::setArg(1)");

    //size_t workgroup_size = getMaxWorkGroupSize(context);
    cl::NDRange globalSize(1);
    cl::NDRange workgroupSize(1);

    if(num_chunks > THREADS_PER_BLOCK) {
        fprintf(stderr, "Using %d chunks of %d bytes, %d threads per block\n", num_chunks, CHUNK_SIZE, THREADS_PER_BLOCK);
        globalSize = cl::NDRange(num_chunks);
        workgroupSize = cl::NDRange(THREADS_PER_BLOCK);
    } 
    
    fprintf(stderr, "enqueueing kernel\n");
    err = queue.enqueueNDRangeKernel(decompressKernel, cl::NullRange, globalSize, workgroupSize);
    checkErr(err, "enqueueNDRangeKernel()");
    checkErr(queue.finish(), "execute kernel");
    return;
}

cl::Buffer CL842Kernels::allocateBuffer(size_t size, cl_mem_flags flags) {
    cl_int err;
    cl::Buffer buffer = cl::Buffer(this->context, flags, size, NULL, &err);
    checkErr(err, "cl::Buffer()");
    return buffer;
}

void CL842Kernels::writeBuffer(cl::Buffer buffer, const void * ptr, size_t size) {
    this->queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size, ptr);
    checkErr(queue.finish(), "enqueueWriteBuffer()");
}

void CL842Kernels::readBuffer(cl::Buffer buffer, void * ptr, size_t size) {
    this->queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, ptr);
    checkErr(queue.finish(), "enqueueReadBuffer()");
}

void CL842Kernels::fillBuffer(cl::Buffer buffer, cl_uint value, size_t offset, size_t size) {
    this->queue.enqueueFillBuffer(buffer, value, offset, size);
    checkErr(queue.finish(), "enqueueFillBuffer()");
}

