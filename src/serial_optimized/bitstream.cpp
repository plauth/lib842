#include <limits.h>
#include <stdlib.h>
#include <stdint.h>

#include "../common/endianness.h"
#include <stdio.h>
//#include <string.h>
#include <inttypes.h>

/* number of bits in a buffered word */
#define wsize 64

/* bit stream structure (opaque to caller) */
struct bitstream {
  uint8_t bits;   /* number of buffered bits (0 <= bits < wsize) */
  uint64_t buffer; /* buffer for incoming/outgoing bits (buffer < 2^bits) */
  uint64_t* ptr;   /* pointer to next uint64_t to be read/written */
  uint64_t* begin; /* beginning of stream */
  uint64_t* end;   /* end of stream (currently unused) */
};

/* private functions ------------------------------------------------------- */

/* read a single uint64_t from memory */
static uint64_t stream_read_word(struct bitstream* s)
{
  uint64_t w = swap_be_to_native64(*s->ptr++);
  return w;
}

/* write a single uint64_t to memory */
static void stream_write_word(struct bitstream* s, uint64_t value)
{
  *s->ptr++ = swap_native_to_be64(value);
}

/* public functions -------------------------------------------------------- */

/* current byte size of stream (if flushed) */
size_t stream_size(const struct bitstream* s)
{
  return sizeof(uint64_t) * (s->ptr - s->begin);
}

/* read 0 <= n <= 64 bits */
uint64_t stream_read_bits(struct bitstream* s, uint8_t n)
{
  uint64_t value = s->buffer >> (wsize - n);

  if (s->bits < n) {
    /* fetch wsize bits  */
    s->buffer = stream_read_word(s);
    value |= s->buffer >> (wsize - (n - s->bits));
    s->buffer <<= n - s->bits;
    s->bits += wsize - n;
    s->buffer *= (s->bits > 0);
  }  else {
    s->bits -= n;
    s->buffer <<= n;
  }
  return value;
}

/* write 0 <= n <= 64 low bits of value and return remaining bits */
void stream_write_bits(struct bitstream* s, uint64_t value, uint8_t n)
{
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
  return ;
}

/* position stream for reading or writing at beginning */
inline void stream_rewind(struct bitstream* s)
{
  s->ptr = s->begin;
  s->buffer = 0;
  s->bits = 0;
}

/* append n zero-bits to stream (n >= 0) */
inline void stream_pad(struct bitstream* s, uint8_t n)
{
  for (s->bits += n; s->bits >= wsize; s->bits -= wsize) {
    stream_write_word(s, s->buffer);
    s->buffer = 0;
  }
}

/* write any remaining buffered bits and align stream on next uint64_t boundary */
void stream_flush(struct bitstream* s)
{
  uint8_t bits = (wsize - s->bits) % wsize;
  if (bits)
    stream_pad(s, bits);
}

/* allocate and initialize bit stream to user-allocated buffer */
struct bitstream* stream_open(void* buffer, size_t bytes)
{
  struct bitstream* s = (struct bitstream*)malloc(sizeof(struct bitstream));
  if (s) {
    s->begin = (uint64_t*)buffer;
    s->end = s->begin + bytes / sizeof(uint64_t);
    stream_rewind(s);
    s->buffer = 0;
  }
  return s;
}

/* close and deallocate bit stream */
void stream_close(struct bitstream* s)
{
  free(s);
}
