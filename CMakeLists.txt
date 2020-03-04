cmake_minimum_required(VERSION 3.0)

option(USE_OPTIMIZED_SERIAL_IMPLEMENTATION "Use optimized sw842_(de)compress implementation" OFF)

project(lib842)

find_package(OpenCL)
if (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
    link_directories(${OpenCL_LIBRARY})
endif()


###########
# LIBRARY #
###########
set(SRCFILES)

# Serial implementation
if (USE_OPTIMIZED_SERIAL_IMPLEMENTATION)
    set(SRCFILES ${SRCFILES} src/serial_optimized/842_compress.cpp
                             src/serial_optimized/842_decompress.cpp
                             src/serial_optimized/bitstream.cpp)
else()
    set(SRCFILES ${SRCFILES} src/serial/842_compress.c
                             src/serial/842_decompress.c)
endif()

# In-kernel, possibly hardware-accelerated implementation (cryptodev)
try_compile(HAVE_CRYPTODEV_LINUX_COMP
            "${CMAKE_BINARY_DIR}/temp"
            "${PROJECT_SOURCE_DIR}/check_cryptodev_linux_comp.c")
if(HAVE_CRYPTODEV_LINUX_COMP)
    set(SRCFILES ${SRCFILES} src/cryptodev/842_interface.c)
endif()

# OpenCL implementation
if (OpenCL_FOUND)
    add_executable(cl2h src/tools/cl2h.cpp)
    add_custom_command(
        OUTPUT src/ocl/decompress.cl.h
        COMMAND cl2h src/ocl/decompress.cl src/ocl/decompress.cl.h CL842_DECOMPRESS_SOURCE
        DEPENDS src/ocl/decompress.cl)

    set(SRCFILES ${SRCFILES} src/ocl/cl842decompress.cpp src/ocl/decompress.cl.h)
endif()

add_library(842 STATIC ${SRCFILES})
set_property(TARGET 842 PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(842 PUBLIC include)
if (OpenCL_FOUND)
    target_link_libraries (842 ${OpenCL_LIBRARY})
endif()

####################
# TESTS / EXAMPLES #
####################
add_executable(test_serial test/compdecomp.c)
target_link_libraries (test_serial 842)

if (HAVE_CRYPTODEV_LINUX_COMP)
    add_executable(test_cryptodev test/compdecomp.c)
    set_target_properties(test_cryptodev PROPERTIES COMPILE_DEFINITIONS "USEHW")
    target_link_libraries (test_cryptodev 842)
endif()

if (OpenCL_FOUND)
    add_executable(test_ocl src/ocl/compdecomp.cpp)
    target_link_libraries (test_ocl 842 ${OpenCL_LIBRARY})

    add_executable(test_ocl_inplace src/ocl/compdecomp.cpp)
    set_target_properties(test_ocl_inplace PROPERTIES COMPILE_DEFINITIONS "INPLACE")
    target_link_libraries (test_ocl_inplace 842 ${OpenCL_LIBRARY})

endif()

# TODOXXX: Integrate all options from the GNU Makefile here and then remove it
