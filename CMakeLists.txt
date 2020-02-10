cmake_minimum_required(VERSION 3.0)

option(USE_OPTIMIZED_SERIAL_IMPLEMENTATION "Use optimized sw842_(de)compress implementation" OFF)

project(lib842)

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

add_library(842 STATIC ${SRCFILES})
set_property(TARGET 842 PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(842 PUBLIC include)

# TODOXXX: Integrate all options from the GNU Makefile here and then remove it
