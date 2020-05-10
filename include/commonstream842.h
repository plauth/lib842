#ifndef __COMMONSTREAM842_H__
#define __COMMONSTREAM842_H__

#ifndef __cplusplus
#error This header is C++-only.
#endif

#include <common842.h>

#include <cstddef>

namespace lib842 {

namespace stream {

static constexpr size_t NUM_CHUNKS_PER_NETWORK_BLOCK = 16;
static constexpr size_t COMPR842_CHUNK_SIZE = 65536;
static constexpr size_t NETWORK_BLOCK_SIZE = NUM_CHUNKS_PER_NETWORK_BLOCK * COMPR842_CHUNK_SIZE;
// This constant must be synchronized with the constant in lib842 (cl842)
// for the integration with OpenCL-based decompression to work
static constexpr size_t COMPRESSIBLE_THRESHOLD = ((COMPR842_CHUNK_SIZE - sizeof(LIB842_COMPRESSED_CHUNK_MARKER) - sizeof(uint64_t)));

} // namespace stream

} // namespace lib842

#endif