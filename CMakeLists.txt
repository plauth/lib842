cmake_minimum_required(VERSION 2.6)

project(dOpenCL)

add_library(842 STATIC src/serial/842_compress.c src/serial/842_decompress.c)
set_property(TARGET 842 PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(842 PUBLIC include)

# TODOXXX: Integrate all options from the GNU Makefile here and then remove it
