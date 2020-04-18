cmake_minimum_required(VERSION 3.1)

project(lib842)

enable_testing()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Avoid 'ignoring attributes on template argument' spam because of types such as std::vector<cl_int>
# See https://github.com/KhronosGroup/OpenCL-CLHPP/issues/37 for more information
if (CMAKE_COMPILER_IS_GNUCXX AND ((CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0) OR (CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 6.0)))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")
endif()

################
# DEPENDENCIES #
################
try_compile(CRYPTODEV_LINUX_COMP_FOUND
            "${CMAKE_BINARY_DIR}/temp"
            "${PROJECT_SOURCE_DIR}/check_cryptodev_linux_comp.c")

include(CheckSymbolExists)
check_symbol_exists(accel_compress "sys/types.h;sys/vminfo.h" AIX_HWCOMPRESSION_FOUND)

find_package(OpenCL)
if (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
    link_directories(${OpenCL_LIBRARY})
endif()
find_package(CUDA)

find_package(OpenMP)
if (OpenMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

###########
# LIBRARY #
###########
set(SRCFILES)

# Serial (unoptimized / fallback) implementation
set(SRCFILES ${SRCFILES} src/serial/842_compress.c
                         src/serial/842_decompress.c)

# Optimized implementation
set(SRCFILES ${SRCFILES} src/serial_optimized/842_compress.cpp
                         src/serial_optimized/842_decompress.cpp
                         src/serial_optimized/bitstream.cpp)

# In-kernel, possibly hardware-accelerated implementation (cryptodev)
if(CRYPTODEV_LINUX_COMP_FOUND)
    set(LIB842_HAVE_CRYPTODEV_LINUX_COMP ON)
    set(SRCFILES ${SRCFILES} src/cryptodev/842_interface.c)
endif()

# OpenCL implementation
if (OpenCL_FOUND)
    set(LIB842_HAVE_OPENCL ON)

    add_executable(cl2c src/tools/cl2c.cpp)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/decompress_cl_842defs.c
        COMMAND cl2c ${CMAKE_CURRENT_SOURCE_DIR}/src/common/842.h ${CMAKE_CURRENT_BINARY_DIR}/decompress_cl_842defs.c CL842_DECOMPRESS_842DEFS_SOURCE
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/common/842.h)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/decompress_cl_kernel.c
        COMMAND cl2c ${CMAKE_CURRENT_SOURCE_DIR}/src/ocl/decompress.cl ${CMAKE_CURRENT_BINARY_DIR}/decompress_cl_kernel.c CL842_DECOMPRESS_KERNEL_SOURCE
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/ocl/decompress.cl)

    set(SRCFILES ${SRCFILES} src/ocl/cl842decompress.cpp decompress_cl_842defs.c decompress_cl_kernel.c)
endif()

# CUDA implementation
if (CUDA_FOUND)
    set(LIB842_HAVE_CUDA ON)
    set(SRCFILES ${SRCFILES} src/cuda/842_decompress.cu)
endif()

configure_file(include/config842.h.in include/config842.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

if (CUDA_FOUND)
    cuda_add_library(842 STATIC ${SRCFILES})
else ()
    add_library(842 STATIC ${SRCFILES})
endif()
set_property(TARGET 842 PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(842 PUBLIC include)
if (OpenCL_FOUND)
    target_link_libraries (842 ${OpenCL_LIBRARY})
endif()

###########
# SAMPLES #
###########
add_executable(test_serial sample/compdecomp.c sample/compdecomp_driver.c)
target_link_libraries (test_serial 842)

add_executable(test_serial_optimized sample/compdecomp.c sample/compdecomp_driver.c)
set_target_properties(test_serial_optimized PROPERTIES COMPILE_DEFINITIONS "USEOPTSW")
target_link_libraries (test_serial_optimized 842)

if (CRYPTODEV_LINUX_COMP_FOUND)
    add_executable(test_cryptodev sample/compdecomp.c sample/compdecomp_driver.c)
    set_target_properties(test_cryptodev PROPERTIES COMPILE_DEFINITIONS "USEHW")
    target_link_libraries (test_cryptodev 842)
endif()

if (AIX_HWCOMPRESSION_FOUND)
    add_executable(test_aix_standalone sample/compdecomp.c sample/compdecomp_driver.c)
    set_target_properties(test_aix_standalone PROPERTIES COMPILE_DEFINITIONS "USEAIX")
endif()

if (CUDA_FOUND)
    cuda_add_executable(test_cuda sample/compdecomp.cu)
    target_link_libraries (test_cuda 842)
endif()

if (OpenCL_FOUND)
    add_executable(test_ocl sample/compdecomp_cl.cpp)
    target_link_libraries (test_ocl 842 ${OpenCL_LIBRARY})

    add_executable(test_ocl_inplace sample/compdecomp_cl.cpp)
    set_target_properties(test_ocl_inplace PROPERTIES COMPILE_DEFINITIONS "USE_INPLACE_COMPRESSED_CHUNKS")
    target_link_libraries (test_ocl_inplace 842 ${OpenCL_LIBRARY})

    add_executable(test_ocl_maybe sample/compdecomp_cl.cpp)
    set_target_properties(test_ocl_maybe PROPERTIES COMPILE_DEFINITIONS "USE_MAYBE_COMPRESSED_CHUNKS")
    target_link_libraries (test_ocl_maybe 842 ${OpenCL_LIBRARY})
endif()

###################
# AUTOMATIC TESTS #
###################
foreach(test test_compress_pattern
             test_compress_pattern_unaligned
             test_compress_pattern_halfsmall
             test_compress_pattern_barelysmall
             test_compress_diffuse
             test_compress_notmultipleof8
             test_decompress_pattern
             test_decompress_pattern_unaligned
             test_decompress_pattern_halfsmall
             test_decompress_pattern_barelysmall
             test_decompress_invalid_simple
             test_decompress_invalid_generator
             test_decompress_crcerror)
    add_executable(${test} test/${test}.c test/test_patterns.c test/test_util.c)
    target_link_libraries (${test} 842)
endforeach()

set(IMPLS_COMPRESS sw optsw)
set(IMPLS_DECOMPRESS sw optsw)
if (CRYPTODEV_LINUX_COMP_FOUND)
    set(IMPLS_COMPRESS ${IMPLS_COMPRESS} hw)
    set(IMPLS_DECOMPRESS ${IMPLS_DECOMPRESS} hw)
endif()
if (OpenCL_FOUND)
    set(IMPLS_DECOMPRESS ${IMPLS_DECOMPRESS} cl)
endif()
set(PATTERNS empty zeros random mixed limit text)

foreach (IMPL ${IMPLS_COMPRESS})
    foreach(PATTERN ${PATTERNS})
        foreach (IMPLD ${IMPLS_DECOMPRESS})
            add_test(test_compress_pattern_${IMPL}_${IMPLD}_${PATTERN} test_compress_pattern ${IMPL} ${IMPLD} ${PATTERN})
        endforeach()
        add_test(test_compress_pattern_unaligned_${IMPL}_${PATTERN} test_compress_pattern_unaligned ${IMPL} ${PATTERN})
        add_test(test_compress_pattern_halfsmall_${IMPL}_${PATTERN} test_compress_pattern_halfsmall ${IMPL} ${PATTERN})
        add_test(test_compress_pattern_barelysmall_${IMPL}_${PATTERN} test_compress_pattern_barelysmall ${IMPL} ${PATTERN})
    endforeach()
    add_test(test_compress_diffuse_${IMPL} test_compress_diffuse ${IMPL})
    add_test(test_compress_notmultipleof8_${IMPL} test_compress_notmultipleof8 ${IMPL})
endforeach()

foreach (IMPL ${IMPLS_DECOMPRESS})
    foreach(PATTERN ${PATTERNS})
        add_test(test_decompress_pattern_${IMPL}_${PATTERN} test_decompress_pattern ${IMPL} ${PATTERN})
        add_test(test_decompress_pattern_unaligned_${IMPL}_${PATTERN} test_decompress_pattern_unaligned ${IMPL} ${PATTERN})
        if (NOT ("${PATTERN}" STREQUAL "empty"))
            add_test(test_decompress_pattern_halfsmall_${IMPL}_${PATTERN} test_decompress_pattern_halfsmall ${IMPL} ${PATTERN})
            add_test(test_decompress_pattern_barelysmall_${IMPL}_${PATTERN} test_decompress_pattern_barelysmall ${IMPL} ${PATTERN})
        endif()
    endforeach()
    add_test(test_decompress_invalid_simple_${IMPL} test_decompress_invalid_simple ${IMPL})
    add_test(test_decompress_invalid_generator_${IMPL} test_decompress_invalid_generator ${IMPL})
    add_test(test_decompress_crcerror_${IMPL} test_decompress_crcerror ${IMPL})
endforeach()
