CXX=g++
CUDAPATH?=/usr/local/cuda

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
all: libclhook.so
endif

COMMONFLAGS=-Wall -fPIC -shared -ldl

libclhook.dylib: clhook.cpp
	$(CXX) $(OPENCL_INC) $(OPENCL_LIB) -Wall -dynamiclib -o libclhook.dylib clhook.cpp

libclhook.so: clhook.cpp
	$(CXX) $(OPENCL_INC) $(OPENCL_LIB) $(COMMONFLAGS) -o libclhook.so cpu_serial/842_compress.c cpu_serial/842_decompress.c clhook.cpp

clean:
	rm -Rf libclhook.so
	rm -Rf libclhook.dylib
