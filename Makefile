CXX=g++
CUDAPATH?=/usr/local/cuda
CUDA_INC?=-I $(CUDAPATH)/include

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
OPENCL_LIB = -framework OpenCL
OPENCL_INC =
all: libclhook.dylib
endif

ifeq ($(UNAME), Linux)
.PHONY: .check-env
.check-env:
	@if [ ! -d "$(CLDIR)" ]; then \
		echo "ERROR: set CLDIR variable."; exit 1; \
	fi
OPENCL_LIB = -L$(CLDIR)/lib -lOpenCL
OPENCL_INC = -I $(CLDIR)/include
all: libclhook.so libcudahook.so
endif

COMMONFLAGS=-Wall -fPIC -shared -ldl

libclhook.dylib: clhook.cpp
	$(CXX) $(OPENCL_INC) $(OPENCL_LIB) -Wall -dynamiclib -o libclhook.dylib clhook.cpp

libclhook.so: clhook.cpp
	$(CXX) $(OPENCL_INC) $(OPENCL_LIB) $(COMMONFLAGS) -o libclhook.so cpu_serial/842_compress.c cpu_serial/842_decompress.c clhook.cpp

libcudahook.so: cudahook.cpp
	$(CXX) $(CUDA_INC) $(COMMONFLAGS) -o libcudahook.so cpu_serial/842_compress.c cpu_serial/842_decompress.c cudahook.cpp

hellocuda: hello.cu
	/usr/local/cuda/bin/nvcc hello.cu -o hello -g --cudart=shared

clean:
	rm -Rf libcudahook.so
	rm -Rf libcudahook.dylib
	rm -Rf libclhook.so
	rm -Rf libclhook.dylib
