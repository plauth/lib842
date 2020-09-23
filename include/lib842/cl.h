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

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <functional>

namespace lib842 {

enum class CLDecompressorInputFormat {
	// This is the simplest format, in which the input buffer contains blocks
	// of size (chunkSize*2), which are always compressed
	// This format is typically not useful for realistic scenarios, due to
	// being suboptimal when dealing with uncompressible (e.g. random) data
	ALWAYS_COMPRESSED_CHUNKS,
	// In this format, the input buffer contains blocks of size chunkSize
	// Inside this buffer, uncompressible data is stored as-is and compressible
	// data is stored with a "marker" header (see LIB842_COMPRESSED_CHUNK_MARKER et al.)
	// This format allows mixing compressed and uncompressed chunks, which minimizes
	// the space overhead. However, uncompressed chunks still need to be copied
	// from the input to the output buffer
	MAYBE_COMPRESSED_CHUNKS,
	// In this format, the input and the output buffer are the same and data is
	// uncompressed in-place in the same buffer.
	//
	// TODOXXX: This mode is broken for some kinds of 'unfortunate' input data.
	// The problem arises in that the output pointer can "catch up" with the
	// input pointer, resulting in corruption or infinite loops.
	// This happens when a chunk contains a lot of redudant data in the beginning
	// (which is encoded as OP_ZEROS/OP_REPEAT/index references in the bitstream)
	// and a lot of uncompressible data at the end (which is encoded as data
	// literals in the bitstream), which causes a bitstreram that requires an
	// 'unbounded' amount of lookahead.
	// So far, no workaround has been found or implemented for this problem.
	// The main alternative is to use the MAYBE_COMPRESSED_CHUNKS mode,
	// which has similar performance but requires additional memory usage
	// and a more complex setup for managing additional temporary buffers
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
	/**
	 * Creates an OpenCL-based decompressor.
	 * Parameters:
	 * - context: OpenCL context over which the decompressor should be usable.
	 * - devices: OpenCL devices over which the decompressor should be usable.
	 * - chunkSize: When decompressing multiple chunks, determines the
	                size in bytes of each chunk of decompressed data.
	 * - chunkStride: When decompressing multiple chunks, determines the
	 *                size in bytes between each chunk of compressed data.
	 * - inputFormat: Determines in which format the input data is passed in.
	 *                See the enumeration for more details.
	 * - error_logger: Returns a stream where errors can be logged.
	 * - debug_logger: Returns a stream where additional information can be logged.
	 */
	CLDeviceDecompressor(const cl::Context &context,
			     const cl::vector<cl::Device> &devices,
			     size_t chunkSize,
			     size_t chunkStride,
			     CLDecompressorInputFormat inputFormat,
			     std::function<std::ostream&(void)> error_logger,
			     std::function<std::ostream&(void)> debug_logger);
	/**
	* Decompress a buffer containing multiple chunks of 842-compressed data.
	* Parameters:
	* - commandQueue: Command queue on which to enqueue the OpenCL operations.
	* - numChunks: Number of chunks to decompress.
	* - inputBuffer: Buffer containing the input data.
	* - inputOffset: Offset in the input buffer at which the compressed data starts.
	*                Must be a multiple of 8.
	* - inputChunkSizes: Specifies the compressed size of each chunk to be decompressed.
	*                    This is only for error-checking, and is optional (can be null).
	* - outputBuffer: Buffer where the output data should be written.
	* - outputOffset: Offset in the output buffer at which the decompressed data should start.
	*                Must be a multiple of 8.
	* - outputChunkSizes: Specifies the maximum size to be written for each chunk.
	*                     This is only for error-checking, and is optional (can be null).
	* - returnValues: The error code (C errno) is written here for each chunk.
	*                 This is only for error-checking, and is optional (can be null).
	* - chunkShuffleMap: Determines the order in which the chunks are processed
	*                    by the OpenCL kernel, which can influence performance
	*                    The reason simply shuffling the order in which chunks are
	*                    processed can help performance is to reduce branch divergence.
	*                    In particular, in the USE_INPLACE_COMPRESSED_CHUNKS
	*                    and USE_MAYBE_COMPRESSED_CHUNKS modes, the decompression kernel
	*                    takes completely different paths for compressible and
	*                    uncompressible chunks. Therefore, shuffling the order in which
	*                    chunks are processed so that all compressible (and respectively
	*                    uncompressible) chunks are processed together can improve
	*                    performance in those cases.
	* - events:          Event wait list before starting decompression
	*                    (like any clEnqueueXXX OpenCL API).
	* - event:           OpenCL event triggered after decompression finishing.
	*                    (like any clEnqueueXXX OpenCL API).
	*/
	void decompress(const cl::CommandQueue &commandQueue,
			size_t numChunks,
			const cl::Buffer &inputBuffer,
			size_t inputOffset,
			const cl::Buffer &inputChunkSizes,
			const cl::Buffer &outputBuffer,
			size_t outputOffset,
			const cl::Buffer &outputChunkSizes,
			const cl::Buffer &returnValues,
			const cl::Buffer &chunkShuffleMap,
			const cl::vector<cl::Event> *events = nullptr,
			cl::Event *event = nullptr) const;

private:
	size_t m_chunkSize;
	size_t m_chunkStride;
	CLDecompressorInputFormat m_inputFormat;
	cl::Program m_program;
	std::function<std::ostream&(void)> m_error_logger;
	std::function<std::ostream&(void)> m_debug_logger;

	void buildProgram(const cl::Context &context, const cl::vector<cl::Device> &devices);
};

/**
 * High-level interface to the OpenCL-based 842 decompressor,
 * for easily compressing data available on the host
 * using any available OpenCL-capable devices.
 */
class CLHostDecompressor
{
public:
	CLHostDecompressor(size_t chunkSize,
			   size_t chunkStride,
			   CLDecompressorInputFormat inputFormat,
			   bool verbose, bool profile);
	void decompress(const uint8_t *input,
			size_t inputBufferSize,
			const size_t *inputChunkSizes,
			uint8_t *output,
			size_t outputBufferSizes,
			size_t *outputChunkSizes,
			size_t *chunkShuffleMap,
			int *returnValues,
			long long *time) const;

private:
	size_t m_chunkSize;
	size_t m_chunkStride;
	CLDecompressorInputFormat m_inputFormat;
	bool m_verbose, m_profile;
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
