//R"=====(
#define OCL
#include "src/ocl/842-internal.h"

/* number of bits in a buffered word */
#define WSIZE 64 //sizeof(uint64_t)

/* rolling fifo sizes */
#define I2_FIFO_SIZE    (2 * (1 << I2_BITS))
#define I4_FIFO_SIZE    (4 * (1 << I4_BITS))
#define I8_FIFO_SIZE    (8 * (1 << I8_BITS))

#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define round_down(x, y) ((x) & ~__round_mask(x, y))

__constant uint16_t fifo_sizes[9] = {
    0,
    0,
    I2_FIFO_SIZE,
    0,
    I4_FIFO_SIZE,
    0,
    0,
    0,
    I8_FIFO_SIZE
};

__constant uint8_t dec_templates[26][4][2] = { // params size in bits
    {OP_DEC_D8, OP_DEC_N0, OP_DEC_N0, OP_DEC_N0}, // 0x00: { D8, N0, N0, N0 }, 64 bits
    {OP_DEC_D4, OP_DEC_D2, OP_DEC_I2, OP_DEC_N0}, // 0x01: { D4, D2, I2, N0 }, 56 bits
    {OP_DEC_D4, OP_DEC_I2, OP_DEC_D2, OP_DEC_N0}, // 0x02: { D4, I2, D2, N0 }, 56 bits
    {OP_DEC_D4, OP_DEC_I2, OP_DEC_I2, OP_DEC_N0}, // 0x03: { D4, I2, I2, N0 }, 48 bits

    {OP_DEC_D4, OP_DEC_I4, OP_DEC_N0, OP_DEC_N0}, // 0x04: { D4, I4, N0, N0 }, 41 bits
    {OP_DEC_D2, OP_DEC_I2, OP_DEC_D4, OP_DEC_N0}, // 0x05: { D2, I2, D4, N0 }, 56 bits
    {OP_DEC_D2, OP_DEC_I2, OP_DEC_D2, OP_DEC_I2}, // 0x06: { D2, I2, D2, I2 }, 48 bits
    {OP_DEC_D2, OP_DEC_I2, OP_DEC_I2, OP_DEC_D2}, // 0x07: { D2, I2, I2, D2 }, 48 bits

    {OP_DEC_D2, OP_DEC_I2, OP_DEC_I2, OP_DEC_I2}, // 0x08: { D2, I2, I2, I2 }, 40 bits
    {OP_DEC_D2, OP_DEC_I2, OP_DEC_I4, OP_DEC_N0}, // 0x09: { D2, I2, I4, N0 }, 33 bits
    {OP_DEC_I2, OP_DEC_D2, OP_DEC_D4, OP_DEC_N0}, // 0x0a: { I2, D2, D4, N0 }, 56 bits
    {OP_DEC_I2, OP_DEC_D4, OP_DEC_I2, OP_DEC_N0}, // 0x0b: { I2, D4, I2, N0 }, 48 bits

    {OP_DEC_I2, OP_DEC_D2, OP_DEC_I2, OP_DEC_D2}, // 0x0c: { I2, D2, I2, D2 }, 48 bits
    {OP_DEC_I2, OP_DEC_D2, OP_DEC_I2, OP_DEC_I2}, // 0x0d: { I2, D2, I2, I2 }, 40 bits
    {OP_DEC_I2, OP_DEC_D2, OP_DEC_I4, OP_DEC_N0}, // 0x0e: { I2, D2, I4, N0 }, 33 bits
    {OP_DEC_I2, OP_DEC_I2, OP_DEC_D4, OP_DEC_N0}, // 0x0f: { I2, I2, D4, N0 }, 48 bits

    {OP_DEC_I2, OP_DEC_I2, OP_DEC_D2, OP_DEC_I2}, // 0x10: { I2, I2, D2, I2 }, 40 bits
    {OP_DEC_I2, OP_DEC_I2, OP_DEC_I2, OP_DEC_D2}, // 0x11: { I2, I2, I2, D2 }, 40 bits
    {OP_DEC_I2, OP_DEC_I2, OP_DEC_I2, OP_DEC_I2}, // 0x12: { I2, I2, I2, I2 }, 32 bits
    {OP_DEC_I2, OP_DEC_I2, OP_DEC_I4, OP_DEC_N0}, // 0x13: { I2, I2, I4, N0 }, 25 bits

    {OP_DEC_I4, OP_DEC_D4, OP_DEC_N0, OP_DEC_N0}, // 0x14: { I4, D4, N0, N0 }, 41 bits
    {OP_DEC_I4, OP_DEC_D2, OP_DEC_I2, OP_DEC_N0}, // 0x15: { I4, D2, I2, N0 }, 33 bits
    {OP_DEC_I4, OP_DEC_I2, OP_DEC_D2, OP_DEC_N0}, // 0x16: { I4, I2, D2, N0 }, 33 bits
    {OP_DEC_I4, OP_DEC_I2, OP_DEC_I2, OP_DEC_N0}, // 0x17: { I4, I2, I2, N0 }, 25 bits


    {OP_DEC_I4, OP_DEC_I4, OP_DEC_N0, OP_DEC_N0}, // 0x18: { I4, I4, N0, N0 }, 18 bits
    {OP_DEC_I8, OP_DEC_N0, OP_DEC_N0, OP_DEC_N0}, // 0x19: { I8, N0, N0, N0 }, 8 bits
};

inline uint64_t bswap(uint64_t value) {
    return
    (uint64_t)((value & (uint64_t)0x00000000000000ffULL) << 56) |
    (uint64_t)((value & (uint64_t)0x000000000000ff00ULL) << 40) |
    (uint64_t)((value & (uint64_t)0x0000000000ff0000ULL) << 24) |
    (uint64_t)((value & (uint64_t)0x00000000ff000000ULL) <<  8) |
    (uint64_t)((value & (uint64_t)0x000000ff00000000ULL) >>  8) |
    (uint64_t)((value & (uint64_t)0x0000ff0000000000ULL) >> 24) |
    (uint64_t)((value & (uint64_t)0x00ff000000000000ULL) >> 40) |
    (uint64_t)((value & (uint64_t)0xff00000000000000ULL) >> 56);
}

inline uint64_t read_bits(struct sw842_param_decomp *p, uint32_t n)
{
  uint64_t value = p->buffer >> (WSIZE - n);
  if (n == 0)
    value = 0;

  if (p->bits < n) {
    p->buffer = bswap(*p->in);
    p->in++;
    value |= p->buffer >> (WSIZE - (n - p->bits));
    p->buffer <<= n - p->bits;
    p->bits += WSIZE - n;
    p->buffer *= (p->bits > 0);
  }  else {
    p->bits -= n;
    p->buffer <<= n;
  }


  return value;
}

inline uint64_t get_index(struct sw842_param_decomp *p, uint8_t size, uint64_t index, uint64_t fsize)
{
    uint64_t offset;
    uint64_t total = round_down(((__global uint8_t*)p->out) - ((__global const uint8_t *)p->ostart), 8);

    offset = index * size;

    /* a ring buffer of fsize is used; correct the offset */
    if (total > fsize) {
        /* this is where the current fifo is */
        uint64_t section = round_down(total, fsize);
        /* the current pos in the fifo */
        uint64_t pos = total - section;

        /* if the offset is past/at the pos, we need to
         * go back to the last fifo section
         */
        if (offset >= pos)
            section -= fsize;

        offset += section;
    }

    return offset;
}

__kernel void decompress(__global uint64_t *in, __global uint64_t *out)
{
    unsigned int chunk_num = get_group_id(0) * get_local_size(0) + get_local_id(0);

    struct sw842_param_decomp p;
    p.ostart = p.out = out + ((CHUNK_SIZE / 8) * chunk_num);
    p.in = (in + ((CHUNK_SIZE / 8 * 2) * chunk_num));


    p.buffer = 0;
    p.bits = 0;

    uint64_t op;

    uint64_t output_word;
    uint32_t bits;
    
    do {
        op = read_bits(&p, OP_BITS);
        output_word = 0;
        bits = 0;

        switch (op) {
            case OP_REPEAT:
                op = read_bits(&p, REPEAT_BITS);
                /* copy op + 1 */
                op++;

                while (op-- > 0) {
                    *p.out = *(p.out -1);
                    p.out++;
                }
                break;
            case OP_ZEROS:
                *p.out = 0;
                p.out++;
                break;
            case OP_END:
                break;
            default:
                for(int i = 0; i < 4; i++) {
                    uint64_t value;

                    uint32_t dec_template = dec_templates[op][i][0];
                    uint32_t is_index = (dec_template >> 7);
                    uint32_t dst_size = dec_templates[op][i][1];

                    value = read_bits(&p, dec_template & 0x7F);

                    if(is_index) {
                        uint64_t offset = get_index(&p, dst_size, value, fifo_sizes[dst_size]);
                        __global uint8_t * ostart8 = (__global uint8_t *) p.ostart;
                        switch(dst_size) {
                            case 2:
                                value = 
                                    (((uint64_t) ostart8[offset    ])) |
                                    (((uint64_t) ostart8[offset + 1]) << 8);
                                break;
                            case 4:
                                value = 
                                    (((uint64_t) ostart8[offset    ])) |
                                    (((uint64_t) ostart8[offset + 1]) << 8 ) | 
                                    (((uint64_t) ostart8[offset + 2]) << 16) |
                                    (((uint64_t) ostart8[offset + 3]) << 24);
                                break;
                            case 8:
                                value = 
                                    (((uint64_t) ostart8[offset    ])) |
                                    (((uint64_t) ostart8[offset + 1]) << 8 ) | 
                                    (((uint64_t) ostart8[offset + 2]) << 16) |
                                    (((uint64_t) ostart8[offset + 3]) << 24) |
                                    (((uint64_t) ostart8[offset + 4]) << 32) |
                                    (((uint64_t) ostart8[offset + 5]) << 40) | 
                                    (((uint64_t) ostart8[offset + 6]) << 48) |
                                    (((uint64_t) ostart8[offset + 7]) << 56);
                                break;
                        }

                        value <<= (WSIZE - (dst_size << 3));
                        value = bswap(value);
                    }
                    output_word |= value << (64 - (dst_size<<3) - bits);
                    bits += dst_size<<3;
                }
                *p.out++ = bswap(output_word);

        }
    } while (op != OP_END);


    return;
}
//)====="