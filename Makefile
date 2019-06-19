CC_FLAGS	:= -Wall -fPIC -std=gnu11 -g -O3 -fopenmp
CXX_FLAGS	:= -Wall -fPIC -std=gnu++11 -g -O3 -fopenmp
NVCC_FLAGS	:= -O3 -maxrregcount 64 -gencode arch=compute_37,code=sm_37 --compile
NVLINKER_FLAGS 	:= --cudart static --relocatable-device-code=false -gencode arch=compute_37,code=compute_37 -gencode arch=compute_37,code=sm_37 -link

NVCC_TEST := $(shell which nvcc 2> /dev/null)
NVCC_AVAILABLE := $(notdir $(NVCC_TEST))

LDFLAGS_OCL := -lOpenCL
CRYPTODEV_IS_LOADED :=

NVCC=nvcc
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
CRYPTODEV_IS_LOADED := $(shell lsmod | grep cryptodev)
else ifeq ($(shell uname -p),ppc)
CXX_FLAGS += -DDISABLE_CRC
else ifeq ($(shell uname -p),s390x)
CXX_FLAGS += -DDISABLE_CRC
else ifeq ($(shell uname -p),x86_64)
CC=gcc
CXX=g++
CRYPTODEV_IS_LOADED := $(shell lsmod | grep cryptodev)
#CC=/opt/intel/compilers_and_libraries/linux/bin/intel64/icc
#CXX=/opt/intel/compilers_and_libraries/linux/bin/intel64/icpc
endif


MODULES   := serial serial_optimized cryptodev aix cuda
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

SRC_DIR_CUDA := src/cuda
OBJ_DIR_CUDA := obj/cuda

SRC_FILES_CUDA := $(wildcard $(SRC_DIR_CUDA)/*.cu)
OBJ_FILES_CUDA := $(patsubst $(SRC_DIR_CUDA)/%.cu,$(OBJ_DIR_CUDA)/%.o,$(SRC_FILES_CUDA))


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
	
$(OBJ_DIR_CUDA)/%.o: $(SRC_DIR_CUDA)/%.cu
	 $(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR_CRYPTODEV)/%.o: $(SRC_DIR_CRYPTODEV)/%.c
	$(CXX) $(CXX_FLAGS) -c $< -o $@

serial_lib: checkdirs $(OBJ_FILES_SERIAL)
	$(CC) $(CC_FLAGS) -shared -Wl,-soname,lib842.so -Wl,--no-as-needed -o bin/serial/lib842.so $(OBJ_FILES_SERIAL)

clean:
	rm -Rf obj
	rm -Rf bin
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
	
test_cuda_standalone: checkdirs $(OBJ_FILES_CUDA)
	$(NVCC) $(OBJ_FILES_CUDA) $(NVLINKER_FLAGS) -o bin/cuda/compdecomp   
	bin/cuda/compdecomp

test_cryptodev: checkdirs $(OBJ_FILES_CRYPTODEV)
	$(CXX) $(CXX_FLAGS) $(OBJ_FILES_CRYPTODEV) -DUSEHW=1 test/compdecomp.c -o bin/cryptodev/compdecomp -I./include 
	bin/cryptodev/compdecomp

goldenunit: checkdirs $(OBJ_FILES_SERIAL_OPT) $(OBJ_FILES_CRYPTODEV)
	$(CXX) $(CXX_FLAGS) $(OBJ_FILES_CRYPTODEV) $(OBJ_FILES_SERIAL_OPT) test/goldenunit1.c -o bin/serial_optimized/goldenunit1 -I./include 
	$(CXX) $(CXX_FLAGS) $(OBJ_FILES_CRYPTODEV) $(OBJ_FILES_SERIAL_OPT) test/goldenunit2.c -o bin/serial_optimized/goldenunit2 -I./include 
	bin/serial_optimized/goldenunit1
	bin/serial_optimized/goldenunit2

ocl:
	$(CXX) $(CXX_FLAGS) src/serial_optimized/842_compress.cpp src/serial_optimized/bitstream.cpp src/ocl/cl842decompress.cpp src/ocl/compdecomp.cpp $(LDFLAGS_OCL) -o ocl_test -I./include

cuda:
	$(NVCC) test/transferbench.cu  -g -O3 obj/serial_optimized/842_decompress.o obj/serial_optimized/842_compress.o -o transferbench -I./include -D_FORCE_INLINES

test_aix_standalone: checkdirs test/compdecomp_aix.c	
	$(CC) -O3 -std=c11 test/compdecomp_aix.c -o bin/aix/compdecomp -Wl,-b64 -maix64 -fopenmp
	LIBPATH=/opt/freeware/lib64 bin/aix/compdecomp

STANDALONE_TARGET := test_serial_standalone test_serial_optimized_standalone

ifneq ($(CRYPTODEV_IS_LOADED),) 
$(info cryptodev kernel module is available, including cryptodev test)
STANDALONE_TARGET += test_cryptodev
endif

ifeq ($(NVCC_AVAILABLE),nvcc)
$(info cuda is available)
STANDALONE_TARGET += test_cuda_standalone
endif

ifeq ($(shell uname -p),ppc)
standalone: test_serial_optimized_standalone
else ifeq ($(shell uname -p),s390x)
standalone: test_serial_optimized_standalone
else
#$(info standlone targets: $(STANDALONE_TARGET))
standalone: $(STANDALONE_TARGET)
endif

libs: serial_lib

test_libs: test_serial_lib

$(BIN_DIR) $(OBJ_DIR):
	@mkdir -p $@
