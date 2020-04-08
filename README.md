# lib842

**lib842** provides efficient, accelerated implementation of the 842 compression algorithm available from userspace. 842 is a compression algorithm by IBM, similar to LZ77, designed for very fast compression and decompression. Thus, it is suitable for cases such as compressed RAM (zram) or I/O link compression.

You can find more information about the algorithm, including a description of its implementation, in our publications:

* [Plauth, Max, and Andreas Polze. "Towards Improving Data Transfer Efficiency for Accelerators Using Hardware Compression." 2018 Sixth International Symposium on Computing and Networking Workshops (CANDARW). IEEE, 2018.](https://ieeexplore.ieee.org/abstract/document/8590885)

* [Plauth, Max, and Andreas Polze. "GPU-Based Decompression for the 842 Algorithm." 2019 Seventh International Symposium on Computing and Networking Workshops (CANDARW). IEEE, 2019.](https://ieeexplore.ieee.org/abstract/document/8951729)

## Implementations

**lib842** provides multiple implementations designed for different accelerators:

* A *simple serial* implementation for the CPU intended to serve as a reference. This implementation is a port of the reference implementation available in the [mainline Linux kernel(https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/lib/842?id=ae46d2aa6a7fbe8ca0946f24b061b6ccdc6c3f25).

* An *optimized serial* implementation for the CPU designed to achieve high compression/decompression.

* A *OpenCL* implementation for OpenCL-capable GPUs designed to achieve high decompression speeds.

* A *CUDA* implementation for CUDA-capable GPUs designed to achieve high decompression speeds.

* A *cryptodev* implementation that allows using the dedicated hardware compressor on IBM POWER7+ hardware.

Currently, the *simple serial*, *optimized serial* and *cryptodev* implementations support both compression and decompression, while the *OpenCL* and *CUDA* implementations support decompression only.

## Requisites

CMake is required in order to execute the build process.

The *simple serial* and *optimized serial* implementations can be built and used on any machine with a modern C/C++ compiler. However, *OpenMP* is needed in order to enable parallel (multithreaded) compression on the CPU.

The *cryptodev* implementation requires building, installing and loading our [modified cryptodev kernel module with compression support](https://github.com/joanbm/cryptodev-linux). On IBM POWER7+ hardware, the dedicated hardware compressor will be automatically used. Otherwise, the Linux kernel will fall back to a non-accelerated implementation.

The *OpenCL* and *CUDA* implementations require a working OpenCL or CUDA environment respectively.

## Getting started

First off, make sure to clone the repository including its submodules (`git clone --recurse-submodules`) in order to pull all required dependencies.

To build lib842, type the following in a shell on the root of the cloned repository:
```
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
```

After a successful compilation, at least the *simple serial* and *optimized serial* implementations will be compiled. You can test compression and decompression by executing one of the samples:
```
./test_serial /path/to/my/file
./test_serial_optimized /path/to/my/file
```

Other implementations will be automatically built if the required dependencies are supported. The build process should provide suitable output to determine which implementations are built or not and why.

You can also verify the implementations are working correctly on your hardware by running the unit tests using the `ctest` command (included with CMake) after a successful build.
