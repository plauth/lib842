FROM ubuntu:bionic

# Install dependencies
RUN apt-get update \
 && apt-get install --yes --no-install-recommends \
        # Build utilities
        cmake make g++ \
        # pocl OpenCL runtime
        pocl-opencl-icd opencl-headers ocl-icd-opencl-dev \
        # Other dependencies
        libnuma-dev \
 && rm -rf /var/lib/apt/lists/*

# Build and test
COPY . /lib842
WORKDIR /lib842/build
RUN cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
CMD ctest
