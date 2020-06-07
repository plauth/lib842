// NOTE: To my best knowledge, this file is based on a bitstream implementation
// from the zpf library (e.g. src/inline/bitstream.c on zpf-0.5.5)
// See: https://computing.llnl.gov/projects/floating-point-compression
// zpf is Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// and licensed under a BSD license

#include <cstdlib>
#include <cstdint>

#include "bitstream.h"
#include "842-internal.h"
#include "../common/endianness.h"

/* number of bits in a buffered word */
#define wsize 64

/* bit stream structure (opaque to caller) */
struct bitstream {
	uint8_t bits; /* number of buffered bits (0 <= bits < wsize) */
	uint64_t buffer; /* buffer for incoming/outgoing bits (buffer < 2^bits) */
	uint64_t *ptr; /* pointer to next uint64_t to be read/written */
	uint64_t *begin; /* beginning of stream */
#ifdef ENABLE_ERROR_HANDLING
	uint64_t *end; /* end of stream */
	bool overfull; /* true after attempting to write past the end of the stream */
#endif
};

/* private functions ------------------------------------------------------- */

/* write a single uint64_t to memory */
static void stream_write_word(struct bitstream *s, uint64_t value)
{
#ifdef ENABLE_ERROR_HANDLING
	if (s->ptr == s->end) {
		s->overfull = true;
		return;
	}
#endif

	*s->ptr++ = swap_native_to_be64(value);
}

/* public functions -------------------------------------------------------- */

#ifdef ENABLE_ERROR_HANDLING
bool stream_is_overfull(const struct bitstream *s)
{
	return s->overfull;
}
#endif

/* current byte size of stream (if flushed) */
size_t stream_size(const struct bitstream *s)
{
	return sizeof(uint64_t) * (s->ptr - s->begin);
}

/* write 0 <= n <= 64 low bits of value and return remaining bits */
void stream_write_bits(struct bitstream *s, uint64_t value, uint8_t n)
{
	// Avoid shift by 64 (only shifts of strictly less bits are allowed by the standard)
	if (n == 0)
		return;
	//shift value with MSB
	value <<= wsize - n;
	/* append bit string to buffer */
	s->buffer |= value >> s->bits;
	s->bits += n;

	/* is buffer full? */
	if (s->bits >= wsize) {
		/* 1 <= n <= 64; decrement n to ensure valid left shifts below */
		value <<= 1;
		n--;

		/* output wsize bits */
		s->bits -= wsize;
		/* assert: 0 <= s->bits <= n */
		stream_write_word(s, s->buffer);
		s->buffer = value << (n - s->bits);
	}
}

/* position stream for reading or writing at beginning */
inline void stream_rewind(struct bitstream *s)
{
	s->ptr = s->begin;
	s->buffer = 0;
	s->bits = 0;
}

/* append n zero-bits to stream (n >= 0) */
inline void stream_pad(struct bitstream *s, uint8_t n)
{
	for (s->bits += n; s->bits >= wsize; s->bits -= wsize) {
		stream_write_word(s, s->buffer);
		s->buffer = 0;
	}
}

/* write any remaining buffered bits and align stream on next uint64_t boundary */
void stream_flush(struct bitstream *s)
{
	uint8_t bits = (wsize - s->bits) % wsize;
	if (bits)
		stream_pad(s, bits);
}

/* allocate and initialize bit stream to user-allocated buffer */
struct bitstream *stream_open(void *buffer, size_t bytes)
{
	auto *s = (struct bitstream *)malloc(sizeof(struct bitstream));
	if (s) {
		s->begin = (uint64_t *)buffer;
#ifdef ENABLE_ERROR_HANDLING
		s->end = s->begin + bytes / sizeof(uint64_t);
		s->overfull = false;
#endif
		stream_rewind(s);
		s->buffer = 0;
	}
	return s;
}

/* close and deallocate bit stream */
void stream_close(struct bitstream *s)
{
	free(s);
}
