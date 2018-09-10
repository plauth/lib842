#include "cl842kernels.hpp"

using namespace std;

std::once_flag CL842Kernels::decompressCompileFlag;
cl::Program CL842Kernels::decompressProg;
cl::Kernel CL842Kernels::decompressKernel;

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
        context = cl::Context(CL_DEVICE_TYPE_CPU, cprops, NULL, NULL, &err);
        checkErr(err, "Context::Context()");
        cl::vector<cl::Device> devices;
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");
        std::string deviceName;
        devices[0].getInfo((cl_device_info)CL_DEVICE_NAME,&deviceName);
        std::cerr << "Device: " << deviceName << std::endl;

        queue = cl::CommandQueue(context, devices[0], 0, &err);
        checkErr(err, "CommandQueue::CommandQueue()");

        prepareDecompressKernel();
}

void CL842Kernels::prepareDecompressKernel() {


    std::call_once(CL842Kernels::decompressCompileFlag, [this]() {
        // load opencl source
        ifstream cl_file("src/ocl/decompress.cl");
        string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));

        CL842Kernels::decompressProg = cl::Program(context, cl_string.c_str());
        cl_int err = decompressProg.build();

        if (err == CL_BUILD_PROGRAM_FAILURE) {
            cl::vector<cl::Device> devices;
            devices = context.getInfo<CL_CONTEXT_DEVICES>();
            checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");
            std::string log = decompressProg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0], NULL);
            std::cerr << "Build Log: " << log << std::endl;
        }
        checkErr(err, "cl::Programm::build()");

        CL842Kernels::decompressKernel = cl::Kernel(CL842Kernels::decompressProg, "decompress", &err);
        checkErr(err, "cl::Kernel()");
    });
}


int CL842Kernels::decompress(cl::Buffer in, uint64_t ilen, cl::Buffer out, cl::Buffer olen) {
    cl_int err;
    err = decompressKernel.setArg(0, in);
    checkErr(err, "Kernel::setArg(0)");
    err = decompressKernel.setArg(1, sizeof(uint64_t), &ilen);
    checkErr(err, "Kernel::setArg(1)");
    err = decompressKernel.setArg(2, out);
    checkErr(err, "Kernel::setArg(2)");
    err = decompressKernel.setArg(3, olen);
    checkErr(err, "Kernel::setArg(3)");

    //size_t workgroup_size = getMaxWorkGroupSize(context);

    err = queue.enqueueNDRangeKernel(decompressKernel, cl::NullRange, cl::NDRange(1,1), cl::NDRange(1, 1));
    checkErr(err, "enqueueNDRangeKernel()");
    return 0;
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

