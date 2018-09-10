#define CHUNK_SIZE 4096

/* special templates */
#define OP_REPEAT   (0x1B)
#define OP_ZEROS    (0x1C)
#define OP_END      (0x1E)

/* additional bits of each op param */
#define OP_BITS     (5)
#define REPEAT_BITS (6)
#define SHORT_DATA_BITS (3)
#define I2_BITS     (8)
#define I4_BITS     (9)
#define I8_BITS     (8)
#define CRC_BITS    (32)

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

/* the max of the regular templates - not including the special templates */
#define OPS_MAX     (0x1a)

#define I2_FIFO_SIZE    (2 * (1 << I2_BITS))
#define I4_FIFO_SIZE    (4 * (1 << I4_BITS))
#define I8_FIFO_SIZE    (8 * (1 << I8_BITS))


#define GENMASK_ULL(h) \
    (~0ULL >> (64 - (h)))

#define __round_mask(y) ((ulong)((y)-1))
#define round_down(x, y) ((x) & ~__round_mask(y))

struct decomp_params {
    __global uchar *in;
    uchar bit;
    ulong ilen;
    __global uchar *out;
    __global uchar *ostart;
    ulong olen;
};

__constant uchar decomp_ops[OPS_MAX][4] = {
    { D8, N0, N0, N0 },
    { D4, D2, I2, N0 },
    { D4, I2, D2, N0 },
    { D4, I2, I2, N0 },
    { D4, I4, N0, N0 },
    { D2, I2, D4, N0 },
    { D2, I2, D2, I2 },
    { D2, I2, I2, D2 },
    { D2, I2, I2, I2 },
    { D2, I2, I4, N0 },
    { I2, D2, D4, N0 },
    { I2, D4, I2, N0 },
    { I2, D2, I2, D2 },
    { I2, D2, I2, I2 },
    { I2, D2, I4, N0 },
    { I2, I2, D4, N0 },
    { I2, I2, D2, I2 },
    { I2, I2, I2, D2 },
    { I2, I2, I2, I2 },
    { I2, I2, I4, N0 },
    { I4, D4, N0, N0 },
    { I4, D2, I2, N0 },
    { I4, I2, D2, N0 },
    { I4, I2, I2, N0 },
    { I4, I4, N0, N0 },
    { I8, N0, N0, N0 }
};

static ushort swap_endianness16(ushort input) {
    return
    ((input & (ushort)0x00ffU) << 8) |
    ((input & (ushort)0xff00U) >> 8);

}

static uint swap_endianness32(uint input) {
    return
    ((input & (uint)0x000000ffUL) << 24) |
    ((input & (uint)0x0000ff00UL) <<  8) |
    ((input & (uint)0x00ff0000UL) >>  8) |
    ((input & (uint)0xff000000UL) >> 24);
}

static ulong swap_endianness64(ulong input) {
    return
    (ulong)((input & (ulong)0x00000000000000ffULL) << 56) |
    (ulong)((input & (ulong)0x000000000000ff00ULL) << 40) |
    (ulong)((input & (ulong)0x0000000000ff0000ULL) << 24) |
    (ulong)((input & (ulong)0x00000000ff000000ULL) <<  8) |
    (ulong)((input & (ulong)0x000000ff00000000ULL) >>  8) |
    (ulong)((input & (ulong)0x0000ff0000000000ULL) >> 24) |
    (ulong)((input & (ulong)0x00ff000000000000ULL) >> 40) |
    (ulong)((input & (ulong)0xff00000000000000ULL) >> 56);
}

typedef union { ushort value16; uint value32; ulong value64; } __attribute__((packed)) unalign;

static ushort read16(__global const void* ptr) { return ((__global const unalign*)ptr)->value16; }
static uint   read32(__global const void* ptr) { return ((__global const unalign*)ptr)->value32; }
static ulong  read64(__global const void* ptr) { return ((__global const unalign*)ptr)->value64; }

static void write16(__global void* ptr, ushort value) { ((__global unalign*)ptr)->value16 = value; }
static void write32(__global void* ptr, uint   value) { ((__global unalign*)ptr)->value32 = value; }
static void write64(__global void* ptr, ulong  value) { ((__global unalign*)ptr)->value64 = value; }

static void __next_bits(struct decomp_params *p, ulong *d, uchar n) {
    __global uchar *in = p->in;
    uchar bits = p->bit + n;

    if (bits <= 8)
        *d = *in >> (8 - bits);
    else if (bits <= 16)
        *d = swap_endianness16(read16(in)) >> (16 - bits);
    else if (bits <= 32)
        *d = swap_endianness32(read32(in)) >> (32 - bits);
    else
        *d = swap_endianness64(read64(in)) >> (64 - bits);

    ulong mask = ((ulong) ~0ULL) >> (64 - n);

    *d &= mask;

    p->bit += n;

    if (p->bit > 7) {
        p->in += p->bit / 8;
        p->ilen -= p->bit / 8;
        p->bit %= 8;
    }
}

static void __split_next_bits(struct decomp_params *p, ulong *d, uchar n, uchar s) {
    ulong tmp = 0;

    __next_bits(p, &tmp, n - s);
    __next_bits(p, d, s);

    *d |= tmp << s;
}

static void next_bits(struct decomp_params *p, ulong *d, uchar n) {
    uchar bits = p->bit + n;

    if (bits > 64) {
        __split_next_bits(p, d, n, 32);
        return;
    } else if (p->ilen < 8 && bits > 32 && bits <= 56) {
        __split_next_bits(p, d, n, 16);
        return;
    } else if (p->ilen < 4 && bits > 16 && bits <= 24) {
        __split_next_bits(p, d, n, 8);
        return;
    }

    __next_bits(p, d, n);
}

static void do_data(struct decomp_params *p, uchar n) {
    ulong v;

    next_bits(p, &v, n * 8);

    switch (n) {
    case 2:
        write16(p->out, swap_endianness16(v));
        break;
    case 4:
        write32(p->out, swap_endianness32(v));
        break;
    case 8:
        write64(p->out, swap_endianness64(v));
        break;
    default:
        return;
    }
    
    p->out += n;
    p->olen -= n;
}

static void __do_index(struct decomp_params *p, uchar size, uchar bits, ulong fsize) {
    ulong index, offset, total = round_down(p->out - p->ostart, 8);
    
    next_bits(p, &index, bits);
    offset = index * size;

    // a ring buffer of fsize is used; correct the offset
    if (total > fsize) {
        // this is where the current fifo is 
        ulong section = round_down(total, fsize);
        /// the current pos in the fifo 
        ulong pos = total - section;

        // if the offset is past/at the pos, we need to
        // go back to the last fifo section
        if (offset >= pos)
            section -= fsize;

        offset += section;
    }

    for(int i = 0; i < size; i++) {
        p->out[0] = p->ostart[offset + 0];
        p->out[1] = p->ostart[offset + 1];
        p->out[2] = p->ostart[offset + 2];
        p->out[3] = p->ostart[offset + 3];
        p->out[4] = p->ostart[offset + 4];
        p->out[5] = p->ostart[offset + 5];
        p->out[6] = p->ostart[offset + 6];
        p->out[7] = p->ostart[offset + 7];

    }
    p->out += size;
    p->olen -= size;
}

static void do_index(struct decomp_params *p, uchar n) {
    switch (n) {
    case 2:
        __do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
        break;
    case 4:
        __do_index(p, 4, I4_BITS, I4_FIFO_SIZE);
        break;
    case 8:
        __do_index(p, 8, I8_BITS, I8_FIFO_SIZE);
        break;
    default:
        break;
    }
}
static void do_op(struct decomp_params *p, uchar o) {
    int i;

    #ifdef DEBUG
    if (o >= OPS_MAX) {
        printf("error: template is %x\n", o);
        return;
    }
    #endif

    for (i = 0; i < 4; i++) {
        uchar op = decomp_ops[o][i];
        switch (op & OP_ACTION) {
        case OP_ACTION_DATA:
            do_data(p, op & OP_AMOUNT);
            break;
        case OP_ACTION_INDEX:
            do_index(p, op & OP_AMOUNT);
            break;
        case OP_ACTION_NOOP:
            break;
        default:
            #ifdef DEBUG
            printf("Internal error, invalid op %x\n", op);
            #endif
            break;
        }
    }
}


__kernel void decompress(__global uchar *in, ulong ilen, __global uchar *out, __global ulong *olen) {
    //_local uchar in_chunk[CHUNK_SIZE];
    //__local uchar out_chunk[CHUNK_SIZE];

    struct decomp_params p;
    ulong op;
    ulong rep, total;

    p.in = in;
    p.bit = 0;
    p.ilen = ilen;
    p.out = out;
    p.ostart = out;
    p.olen = *olen;

    total = p.olen;

    *olen = 0;


    do {
        next_bits(&p, &op, OP_BITS);

        #ifdef DEBUG
        printf("template is %x\n", (uchar)op);
        #endif

        switch ((uchar) op) {
        case OP_REPEAT:
            #ifdef DEBUG
            printf("case(op) = OP_REPEAT\n");
            #endif
            next_bits(&p, &rep, REPEAT_BITS);
            rep++;

            while (rep-- > 0) {
                __global uchar *prev = p.out - 8;
                p.out[0] = prev[0];
                p.out[1] = prev[1];
                p.out[2] = prev[2];
                p.out[3] = prev[3];
                p.out[4] = prev[4];
                p.out[5] = prev[5];
                p.out[6] = prev[6];
                p.out[7] = prev[7];
                p.out += 8;
                p.olen -= 8;
            }

            break;
        case OP_ZEROS:
            #ifdef DEBUG
            printf("case(op) = OP_ZEROS\n");
            #endif
                p.out[0] = 0;
                p.out[1] = 0;
                p.out[2] = 0;
                p.out[3] = 0;
                p.out[4] = 0;
                p.out[5] = 0;
                p.out[6] = 0;
                p.out[7] = 0;
            p.out += 8;
            p.olen -= 8;

            break;
        case 30:
            #ifdef DEBUG
            printf("case(op) = OP_END\n");
            #endif
            break;
        default:
            do_op(&p, (uchar) op);
            
            break;
        }
    } while (op != OP_END);

    *olen = total - p.olen;
}