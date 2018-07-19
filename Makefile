CC_FLAGS	:= -Wall -fPIC -std=gnu11 -g -O3 -fopenmp
CXX_FLAGS	:= -Wall -fPIC -std=gnu++11 -g -O3 -fopt-info-vec=vec.out -Wno-shift-count-overflow -fopenmp

ifeq ($(shell uname),Darwin)
CC=gcc-7
CXX=g++-7
LDFLAGS_OCL := -framework OpenCL
else ifeq ($(shell uname),AIX)
CC=gcc
CXX=g++
CC_FLAGS+=-maix64 -Wl,-b64
else ifeq ($(shell uname -p),ppc64le)
CC=/opt/at11.0/bin/gcc
CXX=/opt/at11.0/bin/g++
NVCC=nvcc
LDFLAGS_OCL := -lOpenCL
else ifeq ($(shell uname -p),x86_64)
CC=gcc
CXX=g++
LDFLAGS_OCL := -lOpenCL
endif




MODULES   := serial serial_optimized cryptodev aix
OBJ_DIR := $(addprefix obj/,$(MODULES))
BIN_DIR := $(addprefix bin/,$(MODULES))

SRC_DIR_SERIAL := src/serial
OBJ_DIR_SERIAL := obj/serial

SRC_FILES_SERIAL := $(wildcard $(SRC_DIR_SERIAL)/*.c)
OBJ_FILES_SERIAL := $(patsubst $(SRC_DIR_SERIAL)/%.c,$(OBJ_DIR_SERIAL)/%.o,$(SRC_FILES_SERIAL))

SRC_DIR_SERIAL_OPT := src/serial_optimized
OBJ_DIR_SERIAL_OPT := obj/serial_optimized

SRC_FILES_SERIAL_OPT := $(wildcard $(SRC_DIR_SERIAL_OPT)/*.cpp)
OBJ_FILES_SERIAL_OPT := $(patsubst $(SRC_DIR_SERIAL_OPT)/%.cpp,$(OBJ_DIR_SERIAL_OPT)/%.o,$(SRC_FILES_SERIAL_OPT))

SRC_DIR_CRYPTODEV := src/cryptodev
OBJ_DIR_CRYPTODEV := obj/cryptodev

SRC_FILES_CRYPTODEV := $(wildcard $(SRC_DIR_CRYPTODEV)/*.c)
OBJ_FILES_CRYPTODEV := $(patsubst $(SRC_DIR_CRYPTODEV)/%.c,$(OBJ_DIR_CRYPTODEV)/%.o,$(SRC_FILES_CRYPTODEV))

.PHONY: all checkdirs clean
#.check-env:

ifeq ($(shell uname),AIX)
all: checkdirs test_aix_standalone
else
all: checkdirs standalone
endif

$(OBJ_DIR_SERIAL)/%.o: $(SRC_DIR_SERIAL)/%.c
	$(CC) $(CC_FLAGS) -c $< -o $@

$(OBJ_DIR_SERIAL_OPT)/%.o: $(SRC_DIR_SERIAL_OPT)/%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(OBJ_DIR_CRYPTODEV)/%.o: $(SRC_DIR_CRYPTODEV)/%.c
	$(CXX) $(CXX_FLAGS) -c $< -o $@

serial_lib: checkdirs $(OBJ_FILES_SERIAL)
	$(CC) $(CC_FLAGS) -shared -Wl,-soname,lib842.so -Wl,--no-as-needed -o bin/serial/lib842.so $(OBJ_FILES_SERIAL)

clean:
	rm -Rf obj
	rm -Rf bin
	rm -Rf test/compdecomp
	rm -Rf vec.out

checkdirs: $(OBJ_DIR) $(BIN_DIR)

test_serial_standalone: checkdirs $(OBJ_FILES_SERIAL)
	$(CC) $(CC_FLAGS) $(OBJ_FILES_SERIAL) test/compdecomp.c -o bin/serial/compdecomp -I./include 
	bin/serial/compdecomp

test_serial_lib: serial_lib
	$(CC) $(CC_FLAGS) test/compdecomp.c -o test/compdecomp -I./include -L./bin/serial/ -l842
	LD_LIBRARY_PATH=$(shell pwd)/$(BIN_DIR):$(shell echo $$LD_LIBRARY_PATH) test/compdecomp

test_serial_optimized_standalone: checkdirs $(OBJ_FILES_SERIAL_OPT)
	$(CXX) $(CXX_FLAGS) $(OBJ_FILES_SERIAL_OPT) test/compdecomp.c -o bin/serial_optimized/compdecomp -I./include 
	bin/serial_optimized/compdecomp

test_cryptodev: checkdirs $(OBJ_FILES_CRYPTODEV)
	$(CXX) $(CXX_FLAGS) $(OBJ_FILES_CRYPTODEV) -DUSEHW=1 test/compdecomp.c -o bin/cryptodev/compdecomp -I./include 
	bin/cryptodev/compdecomp

goldenunit: checkdirs $(OBJ_FILES_SERIAL_OPT) $(OBJ_FILES_CRYPTODEV)
	$(CXX) $(CXX_FLAGS) $(OBJ_FILES_CRYPTODEV) $(OBJ_FILES_SERIAL_OPT) test/goldenunit1.c -o bin/serial_optimized/goldenunit1 -I./include 
	$(CXX) $(CXX_FLAGS) $(OBJ_FILES_CRYPTODEV) $(OBJ_FILES_SERIAL_OPT) test/goldenunit2.c -o bin/serial_optimized/goldenunit2 -I./include 
	bin/serial_optimized/goldenunit1
	bin/serial_optimized/goldenunit2

ocl:
	$(CXX) $(CXX_FLAGS) $(LDFLAGS_OCL) src/ocl/842_decompress.cpp src/ocl/cl842kernels.cpp -o ocl_test -I./include

cuda:
	$(NVCC) test/transferbench.cu  -g -O3 obj/serial_optimized/842_decompress.o obj/serial_optimized/842_compress.o -o transferbench -I./include -D_FORCE_INLINES

test_aix_standalone: checkdirs test/compdecomp_aix.c	
	$(CC) -O3 -std=c11 test/compdecomp_aix.c -o bin/aix/compdecomp -Wl,-b64 -maix64 -fopenmp
	LIBPATH=/opt/freeware/lib64 bin/aix/compdecomp

ifeq ($(shell uname),Darwin)
standalone: test_serial_standalone test_serial_optimized_standalone
else
standalone: test_serial_standalone test_serial_optimized_standalone test_cryptodev
endif


libs: serial_lib

test_libs: test_serial_lib

$(BIN_DIR) $(OBJ_DIR):
	@mkdir -p $@
