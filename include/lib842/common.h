#ifndef LIB842_COMMON_H
#define LIB842_COMMON_H

#include <stdint.h>
#include <stddef.h>

// -----------------------------------------------------------
// Common interface for implementations working in main memory
// -----------------------------------------------------------

// Compresses a sequence of bytes in single chunk mode
// in: Input buffer containing the data to be compressed
// ilen: Size of the input data, in bytes
// out: Output buffer where the uncompressed data will be written
// olen: When called, contains the available size of the output buffer, in bytes
//       On return, contains the size used for compression of the output buffer, in bytes
// Returns: 0 on success, a negative value (errno) on failure
typedef int (*lib842_compress_func)(const uint8_t *in, size_t ilen,
				    uint8_t *out, size_t *olen);

// Decompresses a sequence of bytes in single chunk mode
// in: Input buffer containing the compressed data
// ilen: Size of the input data, in bytes
// out: Output buffer where the original uncompressed data will be written
// olen: When called, contains the available size of the output buffer, in bytes
//       On return, contains the size of the original data, in bytes
// Returns: 0 on success, a negative value (errno) on failure
typedef int (*lib842_decompress_func)(const uint8_t *in, size_t ilen,
				      uint8_t *out, size_t *olen);

// Compresses sequences of bytes in multiple chunk mode
// This is functionally equivalent to calling a lib842_compress_func
// in succession, but may allow better use of the available resources
// (note, however, that it does *not* involve multithreading)
//
// 'isize' and 'osize' are the total sizes of the input and output buffers, where
// chunks are evenly spaced by (isize / numchunks) and (osize / numchunks) bytes
typedef int (*lib842_compress_chunked_func)(
	size_t numchunks,
	const uint8_t *in, size_t isize, const size_t *ilens,
	uint8_t *out, size_t osize, size_t *olens);

// Decompresses sequences of bytes in multiple chunk mode
// This is functionally equivalent to calling a lib842_decompress_func
// in succession, but may allow better use of the available resources
// (note, however, that it does *not* involve multithreading)
//
// 'isize' and 'osize' are the total sizes of the input and output buffers, where
// chunks are evenly spaced by (isize / numchunks) and (osize / numchunks) bytes
typedef int (*lib842_decompress_chunked_func)(
	size_t numchunks,
	const uint8_t *in, size_t isize, const size_t *ilens,
	uint8_t *out, size_t osize, size_t *olens);

struct lib842_implementation {
	lib842_compress_func compress;
	lib842_decompress_func decompress;
	lib842_compress_chunked_func compress_chunked;
	lib842_decompress_chunked_func decompress_chunked;
	size_t alignment;
};

// TODOXXX add prototypes for 'chunked' compression modes,
//         required / preferred alignments

// Used in some contexts where data can be either compressed or decompressed,
// in order to identify which chunks of it are compressed
static const uint8_t LIB842_COMPRESSED_CHUNK_MARKER[16] = {
	0xbe, 0x5a, 0x46, 0xbf, 0x97, 0xe5, 0x2d, 0xd7,
	0xb2, 0x7c, 0x94, 0x1a, 0xee, 0xd6, 0x70, 0x76
};

#endif // LIB842_COMMON_H
