//R"=====(

typedef uchar   uint8_t;
typedef ushort  uint16_t;
typedef short   int16_t;
typedef uint    uint32_t;
typedef ulong   uint64_t; 

/* special templates */
#define OP_REPEAT   (0x1B)
#define OP_ZEROS    (0x1C)
#define OP_END      (0x1E)

/* additional bits of each op param */
#define OP_BITS     (5)
#define REPEAT_BITS (6)
#define I2_BITS     (8)
#define I4_BITS     (9)
#define I8_BITS     (8)
#define D2_BITS     (16)
#define D4_BITS     (32)
#define D8_BITS     (64)
#define CRC_BITS    (32)
#define N0_BITS     (0)

#define REPEAT_BITS_MAX     (0x3f)

/* Arbitrary values used to indicate action */
#define OP_ACTION   (0x70)
#define OP_ACTION_INDEX (0x10)
#define OP_ACTION_DATA  (0x20)
#define OP_ACTION_NOOP  (0x40)
#define OP_AMOUNT   (0x0f)
#define OP_AMOUNT_0 (0x00)
#define OP_AMOUNT_2 (0x02)
#define OP_AMOUNT_4 (0x04)
#define OP_AMOUNT_8 (0x08)

#define D2      (OP_ACTION_DATA  | OP_AMOUNT_2)
#define D4      (OP_ACTION_DATA  | OP_AMOUNT_4)
#define D8      (OP_ACTION_DATA  | OP_AMOUNT_8)
#define I2      (OP_ACTION_INDEX | OP_AMOUNT_2)
#define I4      (OP_ACTION_INDEX | OP_AMOUNT_4)
#define I8      (OP_ACTION_INDEX | OP_AMOUNT_8)
#define N0      (OP_ACTION_NOOP  | OP_AMOUNT_0)

#define DICT16_BITS     (10)
#define DICT32_BITS     (11)
#define DICT64_BITS     (10)

#define I2N (13)
#define I4N (53)
#define I8N (149)

//1st value: position of payload in dataAndIndices
//2nd value: number of bits 
#define D20_OP  {0,  D2_BITS}
#define D21_OP  {1,  D2_BITS}
#define D22_OP  {2,  D2_BITS}
#define D23_OP  {3,  D2_BITS}
#define D40_OP  {4,  D4_BITS}
#define D41_OP  {5,  D4_BITS}
#define D80_OP  {6,  D8_BITS}
#define I20_OP  {7,  I2_BITS}
#define I21_OP  {8,  I2_BITS}
#define I22_OP  {9,  I2_BITS}
#define I23_OP  {10, I2_BITS}
#define I40_OP  {11, I4_BITS}
#define I41_OP  {12, I4_BITS}
#define I80_OP  {13, I8_BITS}
#define D4S_OP  {14, D4_BITS}
#define N0_OP   {15, 0}

#define OP_DEC_NOOP  (0x00)
#define OP_DEC_DATA  (0x00)
#define OP_DEC_INDEX (0x80)

#define OP_DEC_N0   {(N0_BITS | OP_DEC_NOOP),  0}
#define OP_DEC_D2   {(D2_BITS | OP_DEC_DATA),  2}
#define OP_DEC_D4   {(D4_BITS | OP_DEC_DATA),  4}
#define OP_DEC_D8   {(D8_BITS | OP_DEC_DATA),  8}
#define OP_DEC_I2   {(I2_BITS | OP_DEC_INDEX), 2}
#define OP_DEC_I4   {(I4_BITS | OP_DEC_INDEX), 4}
#define OP_DEC_I8   {(I8_BITS | OP_DEC_INDEX), 8}

struct sw842_param_decomp {
    __global uint64_t *out;
    __global const uint64_t* ostart;
    __global const uint64_t *in;
    uint32_t bits;
    uint64_t buffer;
    #ifdef USE_INPLACE_COMPRESSED_CHUNKS
    // FIXME: Determined experimentally. Is this enough for the worst possible case?
    uint64_t lookAheadBuffer[6];
    #endif
};

/* number of bits in a buffered word */
#define WSIZE 64 //sizeof(uint64_t)

/* rolling fifo sizes */
#define I2_FIFO_SIZE    (2 * (1 << I2_BITS))
#define I4_FIFO_SIZE    (4 * (1 << I4_BITS))
#define I8_FIFO_SIZE    (8 * (1 << I8_BITS))

#define __round_mask(x, y) ((y)-1)
#define round_down(x, y) ((x) & ~__round_mask(x, y))

__constant uint16_t fifo_sizes[3] = {
    I2_FIFO_SIZE,
    I4_FIFO_SIZE,
    I8_FIFO_SIZE
};


__constant uint64_t masks[3] = {
    0x000000000000FFFF,
    0x00000000FFFFFFFF,
    0xFFFFFFFFFFFFFFFF
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
    (uint64_t)((value & (uint64_t)0x00000000000000ff) << 56) |
    (uint64_t)((value & (uint64_t)0x000000000000ff00) << 40) |
    (uint64_t)((value & (uint64_t)0x0000000000ff0000) << 24) |
    (uint64_t)((value & (uint64_t)0x00000000ff000000) <<  8) |
    (uint64_t)((value & (uint64_t)0x000000ff00000000) >>  8) |
    (uint64_t)((value & (uint64_t)0x0000ff0000000000) >> 24) |
    (uint64_t)((value & (uint64_t)0x00ff000000000000) >> 40) |
    (uint64_t)((value & (uint64_t)0xff00000000000000) >> 56);
}

inline uint64_t read_bits(struct sw842_param_decomp *p, uint32_t n)
{
  uint64_t value = p->buffer >> (WSIZE - n);
  if (n == 0)
    value = 0;

  if (p->bits < n) {
    #ifdef USE_INPLACE_COMPRESSED_CHUNKS
    p->buffer = p->lookAheadBuffer[0];
    p->lookAheadBuffer[0] = p->lookAheadBuffer[1];
    p->lookAheadBuffer[1] = p->lookAheadBuffer[2];
    p->lookAheadBuffer[2] = p->lookAheadBuffer[3];
    p->lookAheadBuffer[3] = p->lookAheadBuffer[4];
    p->lookAheadBuffer[4] = p->lookAheadBuffer[5];
    p->lookAheadBuffer[5] = bswap(*p->in);
    #else
    p->buffer = bswap(*p->in);
    #endif
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

__kernel void decompress(__global const uint64_t *in, ulong inOffset, __global uint64_t *out, ulong outOffset, ulong numChunks)
{
    unsigned int chunk_num = get_global_id(0);
    if (chunk_num >= numChunks) {
        return;
    }

    struct sw842_param_decomp p;
    p.ostart = p.out = out + (outOffset / 8) + ((CL842_CHUNK_SIZE / 8) * chunk_num);
    p.in = (in + (inOffset / 8) + ((CL842_CHUNK_STRIDE / 8) * chunk_num));

    #if defined(USE_MAYBE_COMPRESSED_CHUNKS) || defined(USE_INPLACE_COMPRESSED_CHUNKS)
    if (p.in[0] != 0xd72de597bf465abe || p.in[1] != 0x7670d6ee1a947cb2) { // = CL842_COMPRESSED_CHUNK_MAGIC
        #ifdef USE_MAYBE_COMPRESSED_CHUNKS
        for (size_t i = 0; i < CL842_CHUNK_SIZE; i++) {
            p.out[i] = p.in[i];
        }
        #endif
        return;
    }
    p.in += (CL842_CHUNK_SIZE - p.in[2]) / 8;
    #endif


    p.buffer = 0;
    #ifdef USE_INPLACE_COMPRESSED_CHUNKS
    p.lookAheadBuffer[0] = bswap(*p.in++);
    p.lookAheadBuffer[1] = bswap(*p.in++);
    p.lookAheadBuffer[2] = bswap(*p.in++);
    p.lookAheadBuffer[3] = bswap(*p.in++);
    p.lookAheadBuffer[4] = bswap(*p.in++);
    p.lookAheadBuffer[5] = bswap(*p.in++);
    #endif
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
                // copy op + 1 
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
                    //printf("op is %x\n", dec_template & 0x7F);
                    uint32_t is_index = (dec_template >> 7);
                    uint32_t dst_size = dec_templates[op][i][1];

                    value = read_bits(&p, dec_template & 0x7F);

                    if(is_index) {
                        uint64_t offset = get_index(&p, dst_size, value, fifo_sizes[dst_size >> 2]);
                        offset >>= 1;
                        __global uint16_t * ostart16 = (__global uint16_t *) p.ostart;
                        value = 
                            (((uint64_t) ostart16[offset    ])) |
                            (((uint64_t) ostart16[offset + 1]) << 16) |
                            (((uint64_t) ostart16[offset + 2]) << 32) |
                            (((uint64_t) ostart16[offset + 3]) << 48);
                        value &= masks[dst_size >> 2];
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
