#include "cl842kernels.hpp"

using namespace std;

CL842Kernels::CL842Kernels() {
        cl_int err;
        cl::vector< cl::Platform > platformList;
        cl::Platform::get(&platformList);
        checkErr(platformList.size()!=0 ? CL_SUCCESS : -1, "cl::Platform::get");
        std::cerr << "Number of available platforms: " << platformList.size() << std::endl;
        cl::Platform platform = platformList[0];

        std::string platformVendor;
        platform.getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);

        std::cerr << "Current platform vendor: " << platformVendor << "\n";
        cl_context_properties cprops[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0};
        //platformList[0]()
        context = cl::Context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, &err);
        checkErr(err, "Context::Context()");
        cl::vector<cl::Device> devices;
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");
        std::string deviceName;
        devices[1].getInfo((cl_device_info)CL_DEVICE_NAME,&deviceName);
        std::cerr << "Device: " << deviceName << std::endl;

        queue = cl::CommandQueue(context, devices[1], 0, &err);
        checkErr(err, "CommandQueue::CommandQueue()");
}

void CL842Kernels::prepareDecompressKernel() {
    std::cout << "Compiling decompress kernel..." << std::endl;
    
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
        printf("Using %d chunks of %d bytes, %d threads per block\n", num_chunks, CHUNK_SIZE, THREADS_PER_BLOCK);
        globalSize = cl::NDRange(num_chunks);
        workgroupSize = cl::NDRange(THREADS_PER_BLOCK);
    } 
    
    printf("enqueueing kernel\n");
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

