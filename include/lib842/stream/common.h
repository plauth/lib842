#ifndef LIB842_STREAM_COMMON_H
#define LIB842_STREAM_COMMON_H

// Common definitions for compression/decompression streams

#ifndef __cplusplus
#error This header is C++-only.
#endif

#include <lib842/common.h>

#include <cstddef>

namespace lib842 {

namespace stream {

// The streams-based compression is based on splitting the
// data to compress on chunks and blocks:
// * A chunk is a block of data, potentially compressed using the 842 algorithm,
//   i.e. either uncompressed data or a single 842-compressed bitstream
// * A block is a set of chunks. The purpose of blocks is to provide a bigger
//   unit to optimize batch transfers, such as transfers through the network
static constexpr size_t CHUNK_SIZE = 65536;
static constexpr size_t NUM_CHUNKS_PER_BLOCK = 16;
static constexpr size_t BLOCK_SIZE = NUM_CHUNKS_PER_BLOCK * CHUNK_SIZE;

// Compressed chunks have a small header, which includes a 'magic' marker
// that makes the chunk recognizable as compressed, and the compressed size
static constexpr size_t MAX_COMPRESSIBLE_THRESHOLD =
	CHUNK_SIZE - sizeof(LIB842_COMPRESSED_CHUNK_MARKER) - sizeof(uint64_t);

// This is the threshold which if surpassed, chunks are kept uncompressed
// (which can make sense to avoid an almost-useless decompression operation)
static constexpr size_t COMPRESSIBLE_THRESHOLD = MAX_COMPRESSIBLE_THRESHOLD;
static_assert(COMPRESSIBLE_THRESHOLD <= MAX_COMPRESSIBLE_THRESHOLD,
	      "COMPRESSIBLE_THRESHOLD <= MAX_COMPRESSIBLE_THRESHOLD");

enum class thread_policy {
	use_defaults,
	spread_threads_among_numa_nodes
};

} // namespace stream

} // namespace lib842

#endif // LIB842_STREAM_COMMON_H
