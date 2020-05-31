//#define USE_CHUNK_SHUFFLE_MAP

#include <iostream>
#include <fstream>
#include <cstdint>
#include <numeric>
#ifdef USE_CHUNK_SHUFFLE_MAP
#include <algorithm>
#endif

#include <lib842/sw.h>
#include <lib842/cl.h>
#include <lib842/common.h>

#include "compdecomp_driver.h"

static constexpr size_t CHUNK_SIZE = 65536;

bool compress_benchmark_core(const uint8_t *in, size_t ilen,
			     size_t *olen,
			     long long *time_comp,
			     long long *time_condense,
			     long long *time_decomp) {
#if defined(USE_INPLACE_COMPRESSED_CHUNKS)
	auto format = lib842::CLDecompressorInputFormat::INPLACE_COMPRESSED_CHUNKS;
#elif defined(USE_MAYBE_COMPRESSED_CHUNKS)
	auto format = lib842::CLDecompressorInputFormat::MAYBE_COMPRESSED_CHUNKS;
#else
	auto format = lib842::CLDecompressorInputFormat::ALWAYS_COMPRESSED_CHUNKS;
#endif

	size_t numChunks = ilen / CHUNK_SIZE;
	size_t chunkStride =
		format == lib842::CLDecompressorInputFormat::ALWAYS_COMPRESSED_CHUNKS
		? CHUNK_SIZE * 2 : CHUNK_SIZE;

	size_t outBufferSize = numChunks * chunkStride;
	size_t decompressedBufferSize = ilen;

	std::vector<uint8_t> out(outBufferSize, 0);
#if defined(USE_INPLACE_COMPRESSED_CHUNKS)
	uint8_t *decompressed = out.data();
#else
	std::vector<uint8_t> decompressedBuf(decompressedBufferSize, 0);
	uint8_t *decompressed = decompressedBuf.data();
#endif

	printf("Using %zu chunks of %zu bytes\n", numChunks, CHUNK_SIZE);

	// -----------
	// COMPRESSION
	// -----------
	std::vector<size_t> chunk_olens(numChunks);
#if defined(USE_INPLACE_COMPRESSED_CHUNKS) || defined(USE_MAYBE_COMPRESSED_CHUNKS)
	/** The objective of (MAYBE|INPLACE)_COMPRESSED_CHUNKS is to define a format that allows
	    easy network transmission of compressed data without excessive copying,
	    buffer overallocation, etc..
	    As part of this, as the name mentions, chunks are decompressed in-place.

	    A file is compressed as follows: It is first chunked into chunks of size
	    CHUNK_SIZE, and each chunk is compressed with 842. Chunks are
	    classified as compressible if the compressed size is
	    <= CHUNK_SIZE - sizeof(LIB842_COMPRESSED_CHUNK_MARKER) - sizeof(uint64_t),
	    otherwise they are considered incompressible.

	    The chunks are placed into the decompression buffer as follows:
	    * For incompressible chunks, the compressed version is thrown away
	      and the uncompressed data is written directly to the buffer.
	    * For compressible chunks, the following is written to the buffer:
	      LIB842_COMPRESSED_CHUNK_MARKER: A sequence of bytes (similar to a UUID)
	      that allows recognizing that this is a compressed chunk.
	    Size (64-bit): The size of the data after compression.
	    *BLANK*: All unused space in the chunk is placed here.
	    Compressed data: All compressed data is placed at the end of the buffer.

	    The (MAYBE|INPLACE)_COMPRESSED_CHUNKS flag is propagated to the OpenCL decompression
	    kernel, which recognizes this format and does a little more work to handle it.
	    Mostly, it ignores uncompressed/incompressible chunks
	    (because it sees that LIB842_COMPRESSED_CHUNK_MARKER is not present),
	    and decompresses compressed chunks in-place, using some lookahead
	    bytes to ensure that the output pointer doesn't 'catch up' the input pointer.
	*/
	static constexpr size_t COMPRESSIBLE_THRESHOLD = 
		CHUNK_SIZE - sizeof(LIB842_COMPRESSED_CHUNK_MARKER) - sizeof(uint64_t);

#pragma omp parallel
	{
		std::vector<uint8_t> temp_buffer(CHUNK_SIZE * 2);

#pragma omp for
		for (size_t chunkNo = 0; chunkNo < numChunks; chunkNo++) {
			const uint8_t *chunk_in = in + (CHUNK_SIZE * chunkNo);
			uint8_t *chunk_out = out.data() + (CHUNK_SIZE * chunkNo);

			chunk_olens[chunkNo] = CHUNK_SIZE * 2;
			optsw842_compress(chunk_in, CHUNK_SIZE, temp_buffer.data(),
					  &chunk_olens[chunkNo]);

			if (chunk_olens[chunkNo] > COMPRESSIBLE_THRESHOLD) {
				memcpy(chunk_out, chunk_in, CHUNK_SIZE);
				continue;
			}

			memcpy(chunk_out, LIB842_COMPRESSED_CHUNK_MARKER,
			       sizeof(LIB842_COMPRESSED_CHUNK_MARKER));
			*reinterpret_cast<uint64_t *>(chunk_out +
				sizeof(LIB842_COMPRESSED_CHUNK_MARKER)) =
				chunk_olens[chunkNo];
			memcpy(&chunk_out[CHUNK_SIZE - chunk_olens[chunkNo]],
			       temp_buffer.data(), chunk_olens[chunkNo]);
		}
	}
#else
#pragma omp parallel for
	for (size_t chunkNo = 0; chunkNo < numChunks; chunkNo++) {
		const uint8_t *chunk_in = in + (CHUNK_SIZE * chunkNo);
		uint8_t *chunk_out = out.data() + ((CHUNK_SIZE * 2) * chunkNo);

		chunk_olens[chunkNo] = CHUNK_SIZE * 2;
		optsw842_compress(chunk_in, CHUNK_SIZE, chunk_out,
				  &chunk_olens[chunkNo]);
	}
#endif

	*olen = std::accumulate(chunk_olens.begin(), chunk_olens.end(), 0);
	*time_comp = -1; // Don't benchmark since compression is not OpenCL-based

	// ------------
	// CONDENSATION
	// ------------
	*time_condense = -1;

	// -------------
	// DECOMPRESSION
	// -------------
#ifdef USE_CHUNK_SHUFFLE_MAP
	std::vector<size_t> chunkShuffleMap(numChunks);
	std::iota(chunkShuffleMap.begin(), chunkShuffleMap.end(), 0);
	std::stable_sort(chunkShuffleMap.begin(), chunkShuffleMap.end(),
		[&chunk_olens](size_t i1, size_t i2) {
		bool uncompressible1 = chunk_olens[i1] >= CHUNK_SIZE;
		bool uncompressible2 = chunk_olens[i2] >= CHUNK_SIZE;
		return uncompressible1 < uncompressible2;
	});
	size_t *chunkShuffleMapPtr = chunkShuffleMap.data();
#else
	size_t *chunkShuffleMapPtr = nullptr;
#endif

	try {
		lib842::CLHostDecompressor clDecompress(
			CHUNK_SIZE, chunkStride,
			format, true, true);
		clDecompress.decompress(out.data(), outBufferSize, nullptr,
					decompressed, decompressedBufferSize, nullptr,
					chunkShuffleMapPtr, nullptr, time_decomp);
	} catch (const cl::Error &ex) {
		fprintf(stderr, "OpenCL error: %s (%d)\n", ex.what(), ex.err());
		return false;
	}

	return memcmp(in, decompressed, ilen) == 0;
}

bool simple_test_core(const uint8_t *in, size_t ilen,
		      uint8_t *out, size_t *olen,
		      uint8_t *decompressed, size_t *dlen)
{
	int err;

	err = optsw842_compress(in, ilen, out, olen);
	if (err != 0) {
		fprintf(stderr, "Error during compression (%d): %s\n",
			-err, strerror(-err));
		return false;
	}

	// TODO: This always uses the
	// CLDecompressorInputFormat::ALWAYS_COMPRESSED_CHUNKS mode
	err = cl842_decompress(out, *olen, decompressed, dlen);
	if (err != 0) {
		fprintf(stderr, "Error during decompression (%d): %s\n",
			-err, strerror(-err));
		return false;
	}

	return true;
}

int main(int argc, const char *argv[])
{
	return compdecomp(argc > 1 ? argv[1] : nullptr, CHUNK_SIZE, 0);
}
