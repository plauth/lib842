#include <iostream>
#include <fstream>
#include <stdint.h>

#include "sw842.h"
#include "cl842.hpp"

using namespace std;

#define STRLEN 32

int main(int argc, char *argv[])
{
	uint8_t *compressIn, *compressOut, *decompressIn, *decompressOut;
	size_t flen, ilen, olen, dlen;

	flen = ilen = olen = dlen = 0;

	size_t num_chunks = 0;

	if (argc <= 1) {
		ilen = CL842HostDecompressor::paddedSize(STRLEN);
	} else if (argc == 2) {
		std::ifstream is(argv[1], std::ifstream::binary);
		if (!is)
			exit(-1);
		is.seekg(0, is.end);
		flen = (size_t)is.tellg();
		is.seekg(0, is.beg);
		is.close();

		ilen = CL842HostDecompressor::paddedSize(flen);

		printf("original file length: %zu\n", flen);
		printf("original file length (padded): %zu\n", ilen);
	}

#if defined(USE_INPLACE_COMPRESSED_CHUNKS) || defined(USE_MAYBE_COMPRESSED_CHUNKS)
	olen = ilen;
#else
	olen = ilen * 2;
#endif
	dlen = ilen;

	compressIn = (uint8_t *)calloc(ilen, sizeof(uint8_t));
	compressOut = (uint8_t *)calloc(olen, sizeof(uint8_t));
	decompressIn = (uint8_t *)calloc(olen, sizeof(uint8_t));
	decompressOut = (uint8_t *)calloc(dlen, sizeof(uint8_t));

	if (argc <= 1) {
		num_chunks = 1;
		uint8_t tmp[] = {
			0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33,
			0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37,
			0x38, 0x38, 0x39, 0x39, 0x40, 0x40, 0x41, 0x41,
			0x42, 0x42, 0x43, 0x43, 0x44, 0x44, 0x45, 0x45
		}; //"0011223344556677889900AABBCCDDEE";
		memcpy(compressIn, tmp, STRLEN);
	} else if (argc == 2) {
		num_chunks = ilen / CL842_CHUNK_SIZE;

		std::ifstream is(argv[1], std::ifstream::binary);
		if (!is)
			exit(-1);
		is.seekg(0, is.beg);
		is.read(reinterpret_cast<char *>(compressIn), flen);

		if (is)
			std::cerr << "successfully read all " << flen
				  << " bytes." << std::endl;
		else
			std::cerr << "error: only " << is.gcount()
				  << "bytes could be read" << std::endl;
		is.close();
	}

	printf("Using %zu chunks of %d bytes\n", num_chunks, CL842_CHUNK_SIZE);

#if defined(USE_INPLACE_COMPRESSED_CHUNKS) || defined(USE_MAYBE_COMPRESSED_CHUNKS)
	/** The objective of (MAYBE|INPLACE)_COMPRESSED_CHUNKS is to define a format that allows
        easy network transmission of compressed data without excessive copying,
        buffer overallocation, etc..
        As part of this, as the name mentions, chunks are decompressed in-place.

        A file is compressed as follows: It is first chunked into chunks of size
        CL842_CHUNK_SIZE, and each chunk is compressed with 842. Chunks are
        classified as compressible if the compressed size is
        <= CL842_CHUNK_SIZE - sizeof(CL842_COMPRESSED_CHUNK_MAGIC) - sizeof(uint64_t),
        otherwise they are considered incompressible.

        The chunks are placed into the decompression buffer as follows:
        * For incompressible chunks, the compressed version is thrown away
          and the uncompressed data is written directly to the buffer.
        * For compressible chunks, the following is written to the buffer:
          CL842_COMPRESSED_CHUNK_MAGIC: A sequence of bytes (similar to a UUID) that allows
                                        recognizing that this is a compressed chunk.
          Size (64-bit): The size of the data after compression.
          *BLANK*: All unused space in the chunk is placed here.
          Compressed data: All compressed data is placed at the end of the buffer.

        The (MAYBE|INPLACE)_COMPRESSED_CHUNKS flag is propagated to the OpenCL decompression kernel,
        which recognizes this format and does a little more work to handle it.
        Mostly, it ignores uncompressed/incompressible chunks
        (because it sees that CL842_COMPRESSED_CHUNK_MAGIC is not present),
        and decompresses compressed chunks in-place, using some lookahead
        bytes to ensure that the output pointer doesn't 'catch up' the input pointer.
    */

#pragma omp parallel
	{
		std::vector<uint8_t> temp_buffer(CL842_CHUNK_SIZE * 2);

#pragma omp for
		for (size_t chunk_num = 0; chunk_num < num_chunks;
		     chunk_num++) {
			size_t chunk_olen = CL842_CHUNK_SIZE * 2;
			uint8_t *chunk_in =
				compressIn + (CL842_CHUNK_SIZE * chunk_num);
			uint8_t *chunk_out =
				compressOut + (CL842_CHUNK_SIZE * chunk_num);

			sw842_compress(chunk_in, CL842_CHUNK_SIZE,
				       &temp_buffer[0], &chunk_olen);
			if (chunk_olen <=
			    CL842_CHUNK_SIZE -
				    sizeof(CL842_COMPRESSED_CHUNK_MAGIC) -
				    sizeof(uint64_t)) {
				memcpy(chunk_out, CL842_COMPRESSED_CHUNK_MAGIC,
				       sizeof(CL842_COMPRESSED_CHUNK_MAGIC));
				*reinterpret_cast<uint64_t *>(
					chunk_out +
					sizeof(CL842_COMPRESSED_CHUNK_MAGIC)) =
					chunk_olen;
				memcpy(&chunk_out[CL842_CHUNK_SIZE - chunk_olen],
				       &temp_buffer[0], chunk_olen);
			} else {
				memcpy(&chunk_out[0], &chunk_in[0],
				       CL842_CHUNK_SIZE);
			}
		}
	}
#else
#pragma omp parallel for
	for (size_t chunk_num = 0; chunk_num < num_chunks; chunk_num++) {
		size_t chunk_olen = CL842_CHUNK_SIZE * 2;
		uint8_t *chunk_in = compressIn + (CL842_CHUNK_SIZE * chunk_num);
		uint8_t *chunk_out =
			compressOut + ((CL842_CHUNK_SIZE * 2) * chunk_num);

		sw842_compress(chunk_in, CL842_CHUNK_SIZE, chunk_out,
			       &chunk_olen);
	}
#endif

	memcpy(decompressIn, compressOut, olen);

	try {
#if defined(USE_INPLACE_COMPRESSED_CHUNKS)
		CL842HostDecompressor clDecompress(
			CL842_CHUNK_SIZE,
			CL842InputFormat::INPLACE_COMPRESSED_CHUNKS, true);
		clDecompress.decompress(decompressIn, olen, decompressIn, dlen);
		memcpy(decompressOut, decompressIn, olen);
#elif defined(USE_MAYBE_COMPRESSED_CHUNKS)
		CL842HostDecompressor clDecompress(
			CL842_CHUNK_SIZE,
			CL842InputFormat::MAYBE_COMPRESSED_CHUNKS, true);
		clDecompress.decompress(decompressIn, olen, decompressOut,
					dlen);
#else
		CL842HostDecompressor clDecompress(
			CL842_CHUNK_SIZE * 2,
			CL842InputFormat::ALWAYS_COMPRESSED_CHUNKS, true);
		clDecompress.decompress(decompressIn, olen, decompressOut,
					dlen);
#endif
	} catch (const cl::Error &ex) {
		std::cerr << "ERROR: " << ex.what() << " (" << ex.err() << ")"
			  << std::endl;
		exit(EXIT_FAILURE);
	}

	if (memcmp(compressIn, decompressOut, ilen) == 0) {
		//printf("Compression performance: %lld ms / %f MiB/s\n", timeend_comp - timestart_comp, (ilen / 1024 / 1024) / ((float) (timeend_comp - timestart_comp) / 1000));
		//printf("Decompression performance: %lld ms / %f MiB/s\n", timeend_decomp - timestart_decomp, (ilen / 1024 / 1024) / ((float) (timeend_decomp - timestart_decomp) / 1000));

		printf("Compression- and decompression was successful!\n");
	} else {
		/*
		for (size_t i = 0; i < ilen; i++) {
			printf("%02x:", compressIn[i]);
		}

		printf("\n\n");

		for (size_t i = 0; i < olen; i++) {
			printf("%02x:", compressOut[i]);
		}

		printf("\n\n");

		for (size_t i = 0; i < dlen; i++) {
			printf("%02x:", decompressOut[i]);
		}
		*/

		printf("\n\n");

		fprintf(stderr,
			"FAIL: Decompressed data differs from the original input data.\n");
	}

	return 0;
}
