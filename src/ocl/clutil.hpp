#ifndef CLUTIL_HPP
#define CLUTIL_HPP

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <iostream>
#include "cl2.hpp"

inline void checkErr(cl_int err, const char * name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name  << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline size_t getMaxWorkGroupSize(cl::Context context) {
    size_t max_wg_size = SIZE_MAX;
    size_t max_x_dim = SIZE_MAX;
    size_t max_y_dim = SIZE_MAX;
    size_t max_z_dim = SIZE_MAX;

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (devices.size() > 0) {
        for (size_t i = 0; i < devices.size(); i++) {
            max_wg_size = devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() < max_wg_size? devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() : max_wg_size;
            std::vector<size_t> itemSizes = devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            switch(devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>()) {
            case 3:
                max_z_dim = itemSizes[2] < max_z_dim ? itemSizes[2] : max_z_dim;
            case 2:
                max_y_dim = itemSizes[1] < max_y_dim ? itemSizes[1] : max_y_dim;
            case 1:
                max_x_dim = itemSizes[1] < max_x_dim ? itemSizes[1] : max_x_dim;
            }
        }
    }

    return std::min(max_wg_size,max_x_dim);
}

#endif // CLUTIL_HPP
