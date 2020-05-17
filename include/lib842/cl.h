#ifndef LIB842_CL_H
#define LIB842_CL_H

#include <lib842/config.h>

#ifdef LIB842_HAVE_OPENCL

#ifdef __cplusplus

#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <stddef.h>
#include <stdint.h>

namespace lib842 {

enum class CLDecompressorInputFormat {
	// This is the simplest format, in which the input buffer contains blocks
	// of size (inputChunkSize*2), which are always compressed
	// This format is typically not useful for realistic scenarios, due to
	// being suboptimal when dealing with uncompressible (e.g. random) data
	ALWAYS_COMPRESSED_CHUNKS,
	// In this format, the input buffer contains blocks of size inputChunkSize
	// Inside this buffer, uncompressible data is stored as-is and compressible
	// data is stored with a "marker" header (see LIB842_COMPRESSED_CHUNK_MARKER et al.)
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
 * Low-level interface to the OpenCL-based 842 decompressor,
 * for integration into existing OpenCL applications
 * where context, command queue, buffers, etc. are already available.
 */
class CLDeviceDecompressor
{
	public:
		CLDeviceDecompressor(const cl::Context& context,
					const cl::vector<cl::Device>& devices,
					size_t inputChunkSize,
					size_t inputChunkStride,
					CLDecompressorInputFormat inputFormat,
					bool verbose = false);
		void decompress(const cl::CommandQueue& commandQueue,
				const cl::Buffer& inputBuffer, size_t inputOffset,
				size_t inputSize, const cl::Buffer &inputSizes,
				const cl::Buffer& outputBuffer, size_t outputOffset,
				size_t outputSize, const cl::Buffer &outputSizes,
				const cl::Buffer &returnValues,
				const cl::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr) const;

	private:
		size_t m_inputChunkSize;
		size_t m_inputChunkStride;
		CLDecompressorInputFormat m_inputFormat;
		bool m_verbose;
		cl::Program m_program;

		void buildProgram(const cl::Context& context, const cl::vector<cl::Device>& devices);
};

/**
 * High-level interface to the OpenCL-based 842 decompressor,
 * for easily compressing data available on the host
 * using any available OpenCL-capable devices.
 */
class CLHostDecompressor
{
	public:
		CLHostDecompressor(size_t inputChunkSize,
				   size_t inputChunkStride,
				   CLDecompressorInputFormat inputFormat,
				   bool verbose = false);
		void decompress(const uint8_t* input, size_t inputSize,
				const size_t *inputSizes,
				uint8_t* output, size_t outputSize,
				size_t *outputSizes, int *returnValues) const;

	private:
		size_t m_inputChunkStride;
		CLDecompressorInputFormat m_inputFormat;
		bool m_verbose;
		cl::vector<cl::Device> m_devices;
		cl::Context m_context;
		cl::CommandQueue m_queue;
		CLDeviceDecompressor m_deviceCompressor;

		cl::vector<cl::Device> findDevices() const;
};

} // namespace lib842

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

#endif // LIB842_CL_H
