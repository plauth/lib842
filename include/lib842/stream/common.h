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

static constexpr size_t NUM_CHUNKS_PER_NETWORK_BLOCK = 16;
static constexpr size_t COMPR842_CHUNK_SIZE = 65536;
static constexpr size_t NETWORK_BLOCK_SIZE = NUM_CHUNKS_PER_NETWORK_BLOCK * COMPR842_CHUNK_SIZE;
static constexpr size_t COMPRESSIBLE_THRESHOLD = ((COMPR842_CHUNK_SIZE - sizeof(LIB842_COMPRESSED_CHUNK_MARKER) - sizeof(uint64_t)));

} // namespace stream

} // namespace lib842

#endif // LIB842_STREAM_COMMON_H
