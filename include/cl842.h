#ifndef __CL842_H__
#define __CL842_H__

#include "config842.h"

#ifdef LIB842_HAVE_OPENCL

#ifdef __cplusplus

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <stddef.h>
#include <stdint.h>

#define CL842_CHUNK_SIZE 65536

static const uint8_t CL842_COMPRESSED_CHUNK_MAGIC[16] = {
	0xbe, 0x5a, 0x46, 0xbf, 0x97, 0xe5, 0x2d, 0xd7, 0xb2, 0x7c, 0x94, 0x1a, 0xee, 0xd6, 0x70, 0x76
};

enum class CL842InputFormat {
	// This is the simplest format, in which the input buffer contains blocks
	// of size (CL842_CHUNK_SIZE*2), which are always compressed
	// This format is typically not useful for realistic scenarios, due to
	// being suboptimal when dealing with uncompressible (e.g. random) data
	ALWAYS_COMPRESSED_CHUNKS,
	// In this format, the input buffer contains blocks of size CL842_CHUNK_SIZE
	// Inside this buffer, uncompressible data is stored as-is and compressible
	// data is stored with a "marker" header (see CL842_COMPRESSED_CHUNK_MAGIC et al.)
	// This format allows mixing compressed and uncompressed chunks, which minimizes
	// the space overhead. However, uncompressed chunks still need to be copied
	// from the input to the output buffer
	MAYBE_COMPRESSED_CHUNKS,
	// In this format, the input and the output buffer are the same and data is
	// uncompressed in-place in the same buffer. However, this fails with some
	// data where the output pointer "catches up" with the input pointer
	// (TODOXXX: Fix or delete this mode)
	INPLACE_COMPRESSED_CHUNKS,
};

/**
 * Low-level interface to CL842, for integration into existing OpenCL applications
 * where context, command queue, buffers, etc. are already available.
 */
class CL842DeviceDecompressor
{
	public:
		CL842DeviceDecompressor(const cl::Context& context,
					const VECTOR_CLASS<cl::Device>& devices,
					size_t inputChunkStride,
					CL842InputFormat inputFormat,
					bool verbose = false);
		void decompress(const cl::CommandQueue& commandQueue,
				const cl::Buffer& inputBuffer, size_t inputOffset,
				size_t inputSize, const cl::Buffer &inputSizes,
				const cl::Buffer& outputBuffer, size_t outputOffset,
				size_t outputSize, const cl::Buffer &outputSizes,
				const cl::Buffer &returnValues,
				const VECTOR_CLASS<cl::Event>* events = nullptr, cl::Event* event = nullptr);

	private:
		size_t m_inputChunkStride;
		CL842InputFormat m_inputFormat;
		bool m_verbose;
		cl::Program m_program;

		void buildProgram(const cl::Context& context, const VECTOR_CLASS<cl::Device>& devices);
};

/**
 * High-level interface to CL842, for easily compressing data available on the host
 * using any available OpenCL-capable devices.
 */
class CL842HostDecompressor
{
	public:
		static size_t paddedSize(size_t size);

		CL842HostDecompressor(size_t inputChunkStride,
				      CL842InputFormat inputFormat,
				      bool verbose = false);
		void decompress(const uint8_t* input, size_t inputSize,
				const size_t *inputSizes,
				uint8_t* output, size_t outputSize,
				size_t *outputSizes, int *returnValues);

	private:
		size_t m_inputChunkStride;
		CL842InputFormat m_inputFormat;
		bool m_verbose;
		VECTOR_CLASS<cl::Device> m_devices;
		cl::Context m_context;
		cl::CommandQueue m_queue;
		CL842DeviceDecompressor m_deviceCompressor;

		VECTOR_CLASS<cl::Device> findDevices();
};

extern "C" {
#endif

// WARNING: This simple interface is intended for testing only
// It will only use 1 GPU thread (not decompress chunks in parallel)
int cl842_decompress(const uint8_t *in, size_t ilen,
		     uint8_t *out, size_t *olen);


#ifdef __cplusplus
}
#endif

#endif

#endif
