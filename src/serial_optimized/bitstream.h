// NOTE: To my best knowledge, this file is based on a bitstream implementation
// from the zpf library (e.g. src/inline/bitstream.c on zpf-0.5.5)
// See: https://computing.llnl.gov/projects/floating-point-compression
// zpf is Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// and licensed under a BSD license

#ifndef __BITSTREAM_H__
#define __BITSTREAM_H__

struct bitstream *stream_open(void *buffer, size_t bytes);
void stream_close(struct bitstream *s);
size_t stream_size(const struct bitstream *s);
void stream_write_bits(struct bitstream *s, uint64_t value, uint8_t n);
size_t stream_flush(struct bitstream *s);

#endif
