#include <limits.h>
#include <stdlib.h>
#include <stdint.h>

#include "../common/endianness.h"
//#include <stdio.h>
//#include <string.h>
//#include <inttypes.h>

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
  uint64_t w = *s->ptr++;
  return w;
}

/* write a single uint64_t to memory */
static void stream_write_word(struct bitstream* s, uint64_t value)
{
  *s->ptr++ = swap_endianness64(value);
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
  //TODO: stream_read_bits is not yet ported to MSB-first bitstream layout, hence it will return garbage!
  uint64_t value = s->buffer;
  if (s->bits < n) {
    /* keep fetching wsize bits until enough bits are buffered */
    do {
      /* assert: 0 <= s->bits < n <= 64 */
      s->buffer = stream_read_word(s);
      value += (uint64_t)s->buffer << s->bits;
      s->bits += wsize;
    } while (sizeof(s->buffer) < sizeof(value) && s->bits < n);
    /* assert: 1 <= n <= s->bits < n + wsize */
    s->bits -= n;
    if (!s->bits) {
      /* value holds exactly n bits; no need for masking */
      s->buffer = 0;
    }
    else {
      /* assert: 1 <= s->bits < wsize */
      s->buffer >>= wsize - s->bits;
      /* assert: 1 <= n <= 64 */
      value &= ((uint64_t)2 << (n - 1)) - 1;
    }
  }
  else {
    /* assert: 0 <= n <= s->bits < wsize <= 64 */
    s->bits -= n;
    s->buffer >>= n;
    value &= ((uint64_t)1 << n) - 1;
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
  }
  return s;
}

/* close and deallocate bit stream */
void stream_close(struct bitstream* s)
{
  free(s);
}

/*
int main( int argc, const char* argv[])
{
  uint8_t *buffer = (uint8_t *) malloc(32);
  memset(buffer, 0, 32);
  struct bitstream* stream = stream_open(buffer, 32);
  printf("wsize = %d\n", wsize);

  uint8_t tmp1[] = {0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x10, 0x10};
  uint8_t tmp2[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0x05};
  uint8_t tmp3[] = {0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x28, 0x00};

  uint64_t val1, val2, val3;

  memcpy(&val1, tmp1, 8);
  memcpy(&val2, tmp2, 8);
  memcpy(&val3, tmp3, 8);


  stream_write_bits(stream, (uint64_t) 0x0000000000000000, 5);
  stream_write_bits(stream, swap_endianness64(val1), 64);
  stream_write_bits(stream, (uint64_t) 0x0000000000000011, 5);
  stream_write_bits(stream, swap_endianness64(val2), 40);
  stream_write_bits(stream, (uint64_t) 0x000000000000000c, 5);
  stream_write_bits(stream, swap_endianness64(val3), 48);
 

  stream_flush(stream);

  for (int i = 0; i < 32; i++) {
    printf("%02x:", buffer[i]);
  }

  printf("\n");

}*/
