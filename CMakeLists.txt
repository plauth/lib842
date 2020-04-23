cmake_minimum_required(VERSION 3.1)

project(lib842)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

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
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

try_compile(CRYPTODEV_LINUX_COMP_FOUND
            "${CMAKE_BINARY_DIR}/temp"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_cryptodev_linux_comp.c")

include(CheckSymbolExists)
check_symbol_exists(accel_compress "sys/types.h;sys/vminfo.h" AIX_HWCOMPRESSION_FOUND)

find_package(OpenCL)
find_package(CUDA)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)
find_package(OpenMP)
# Fix for CMake < 3.9: https://web.archive.org/web/20190817053313/https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
if(NOT TARGET OpenMP::OpenMP_C)
    add_library(OpenMP::OpenMP_C IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_C PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_C_FLAGS})
    set_property(TARGET OpenMP::OpenMP_C PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_C_FLAGS} Threads::Threads)
endif()
if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
endif()

find_package(NUMA)

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
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/decompress_cl_842defs.c"
        COMMAND cl2c "${CMAKE_CURRENT_SOURCE_DIR}/src/common/842.h" "${CMAKE_CURRENT_BINARY_DIR}/decompress_cl_842defs.c" CL842_DECOMPRESS_842DEFS_SOURCE
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/common/842.h")
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/decompress_cl_kernel.c"
        COMMAND cl2c "${CMAKE_CURRENT_SOURCE_DIR}/src/ocl/decompress.cl" "${CMAKE_CURRENT_BINARY_DIR}/decompress_cl_kernel.c" CL842_DECOMPRESS_KERNEL_SOURCE
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/ocl/decompress.cl")

    set(SRCFILES ${SRCFILES} src/ocl/cl842decompress.cpp decompress_cl_842defs.c decompress_cl_kernel.c)
endif()

# CUDA implementation
if (CUDA_FOUND)
    set(LIB842_HAVE_CUDA ON)
    set(SRCFILES ${SRCFILES} src/cuda/842_decompress.cu)
endif()

configure_file(include/config842.h.in include/config842.h @ONLY)

if (CUDA_FOUND)
    # NB: This is necessary for the CUDA build to work on CMake 3.5.1,
    #     as it appears that (target_)include_directories is not respected there
    cuda_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}/include")

    cuda_add_library(842 STATIC ${SRCFILES})
else ()
    add_library(842 STATIC ${SRCFILES})
endif()
set_property(TARGET 842 PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(842 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}/include")
if (OpenCL_FOUND)
    target_include_directories(842 PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(842 ${OpenCL_LIBRARIES})
endif()

###########
# SAMPLES #
###########
add_executable(test_serial sample/compdecomp.c sample/compdecomp_driver.c)
target_link_libraries(test_serial 842)
if (OpenMP_FOUND)
    target_link_libraries(test_serial OpenMP::OpenMP_C)
endif()

add_executable(test_serial_optimized sample/compdecomp.c sample/compdecomp_driver.c)
target_compile_definitions(test_serial_optimized PRIVATE "USEOPTSW")
target_link_libraries(test_serial_optimized 842)
if (OpenMP_FOUND)
    target_link_libraries(test_serial_optimized OpenMP::OpenMP_C)
endif()

add_executable(test_serial_optimized_cppthreads sample/compdecomp_cppthreads.cpp sample/compdecomp_driver.c)
target_compile_definitions(test_serial_optimized_cppthreads PRIVATE "USEOPTSW")
target_link_libraries(test_serial_optimized_cppthreads 842 Threads::Threads)
if (NUMA_FOUND)
    target_compile_definitions(test_serial_optimized_cppthreads PRIVATE "SPREAD_THREADS_AMONG_NUMA_NODES")
    target_link_libraries(test_serial_optimized_cppthreads NUMA::NUMA)
endif()

if (CRYPTODEV_LINUX_COMP_FOUND)
    add_executable(test_cryptodev sample/compdecomp.c sample/compdecomp_driver.c)
    target_compile_definitions(test_cryptodev PRIVATE "USEHW")
    target_link_libraries(test_cryptodev 842)
    if (OpenMP_FOUND)
        target_link_libraries(test_cryptodev OpenMP::OpenMP_C)
    endif()

    add_executable(test_cryptodev_cppthreads sample/compdecomp_cppthreads.cpp sample/compdecomp_driver.c)
    target_compile_definitions(test_cryptodev_cppthreads PRIVATE "USEHW")
    target_link_libraries(test_cryptodev_cppthreads 842 Threads::Threads)
    if (NUMA_FOUND)
        target_compile_definitions(test_cryptodev_cppthreads PRIVATE "SPREAD_THREADS_AMONG_NUMA_NODES")
        target_link_libraries(test_cryptodev_cppthreads NUMA::NUMA)
    endif()
endif()

if (AIX_HWCOMPRESSION_FOUND)
    add_executable(test_aix_standalone sample/compdecomp.c sample/compdecomp_driver.c)
    target_compile_definitions(test_aix_standalone PRIVATE "USEAIX")
    if (OpenMP_FOUND)
        target_link_libraries(test_aix_standalone OpenMP::OpenMP_C)
    endif()
endif()

if (CUDA_FOUND)
    # Hacky to mix OpenMP with CUDA (using FindCUDA)
    # Should get cleaner with CMake 3.10+ and CUDA as a first-class language
    set(OpenMP_CUDA_FLAGS "")
    if (OpenMP_FOUND)
        foreach (FLAG ${OpenMP_CXX_FLAGS})
            set(OpenMP_CUDA_FLAGS "${OpenMP_CUDA_FLAGS}--compiler-options;${FLAG};")
        endforeach()
    endif()
    cuda_add_executable(test_cuda sample/compdecomp.cu OPTIONS ${OpenMP_CUDA_FLAGS})
    target_link_libraries(test_cuda 842)
    target_link_libraries(test_cuda ${OpenMP_CXX_FLAGS})
endif()

if (OpenCL_FOUND)
    add_executable(test_ocl sample/compdecomp_cl.cpp)
    target_link_libraries(test_ocl 842 ${OpenCL_LIBRARIES})
    target_include_directories(test_ocl PRIVATE ${OpenCL_INCLUDE_DIRS})
    if (OpenMP_FOUND)
        target_link_libraries(test_ocl OpenMP::OpenMP_CXX)
    endif()

    add_executable(test_ocl_inplace sample/compdecomp_cl.cpp)
    target_compile_definitions(test_ocl_inplace PRIVATE "USE_INPLACE_COMPRESSED_CHUNKS")
    target_link_libraries(test_ocl_inplace 842 ${OpenCL_LIBRARIES})
    target_include_directories(test_ocl_inplace PRIVATE ${OpenCL_INCLUDE_DIRS})
    if (OpenMP_FOUND)
        target_link_libraries(test_ocl_inplace OpenMP::OpenMP_CXX)
    endif()

    add_executable(test_ocl_maybe sample/compdecomp_cl.cpp)
    target_compile_definitions(test_ocl_maybe PRIVATE "USE_MAYBE_COMPRESSED_CHUNKS")
    target_link_libraries(test_ocl_maybe 842 ${OpenCL_LIBRARIES})
    target_include_directories(test_ocl_maybe PRIVATE ${OpenCL_INCLUDE_DIRS})
    if (OpenMP_FOUND)
        target_link_libraries(test_ocl_maybe OpenMP::OpenMP_CXX)
    endif()
endif()

###################
# AUTOMATIC TESTS #
###################
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
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
        target_link_libraries(${test} 842)
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
endif()
