// NB: When we build the program, we prepend the 842 definitions,
//     i.e. we kind of do the equivalent to this include:
// #include "../common/842.h"

typedef uchar uint8_t;
typedef ushort uint16_t;
typedef short int16_t;
typedef uint uint32_t;
typedef ulong uint64_t;

// Define NULL, since it is not required by the OpenCL C 1.2 standard
// Most common vendor implementations define it anyway (Intel, NVIDIA),
// but pocl adheres strictly to the standard and doesn't
// See also: https://github.com/pocl/pocl/issues/831
#ifndef NULL
#define NULL 0L
#endif

#if defined(USE_MAYBE_COMPRESSED_CHUNKS) || defined(USE_INPLACE_COMPRESSED_CHUNKS)
__constant static const uint8_t LIB842_COMPRESSED_CHUNK_MARKER[] =
	LIB842_COMPRESSED_CHUNK_MARKER_DEF; // Defined at build time
#endif

#ifndef USE_INPLACE_COMPRESSED_CHUNKS
#define RESTRICT_UNLESS_INPLACE restrict
#else
#define RESTRICT_UNLESS_INPLACE
#endif

#define ENABLE_ERROR_HANDLING

struct sw842_param_decomp {
	__global uint64_t *out;
	__global const uint64_t *ostart;
	__global const uint64_t *in;
#ifdef ENABLE_ERROR_HANDLING
	__global const uint64_t *istart;
	size_t ilen;
	size_t olen;
	int errorcode;
#endif
	uint32_t bits;
	uint64_t buffer;
#ifdef USE_INPLACE_COMPRESSED_CHUNKS
	// TODOXXX: This amount of lookahead is insufficient, and can be overflowed
	// on certain 'unfortunate' cases of input data.
	// This causes this mode to be currently 'broken' for the general case
	// See the notes in the comments on cl.h for more details
	uint64_t lookAheadBuffer[6];
#endif
};

/* number of bits in a buffered word */
#define WSIZE 64 //sizeof(uint64_t)

/* rolling fifo sizes */
#define I2_FIFO_SIZE (2 * (1 << I2_BITS))
#define I4_FIFO_SIZE (4 * (1 << I4_BITS))
#define I8_FIFO_SIZE (8 * (1 << I8_BITS))

#define __round_mask(x, y) ((y)-1)
#define round_down(x, y) ((x) & ~__round_mask(x, y))

__constant static const uint16_t fifo_sizes[3] = { I2_FIFO_SIZE, I4_FIFO_SIZE, I8_FIFO_SIZE };
__constant static const uint64_t masks[3] = { 0x000000000000FFFF, 0x00000000FFFFFFFF, 0xFFFFFFFFFFFFFFFF };
__constant static const uint8_t dec_templates[26][4][2] = {
	// params size in bits
	{ OP_DEC_D8, OP_DEC_N0, OP_DEC_N0, OP_DEC_N0 }, // 0x00: { D8, N0, N0, N0 }, 64 bits
	{ OP_DEC_D4, OP_DEC_D2, OP_DEC_I2, OP_DEC_N0 }, // 0x01: { D4, D2, I2, N0 }, 56 bits
	{ OP_DEC_D4, OP_DEC_I2, OP_DEC_D2, OP_DEC_N0 }, // 0x02: { D4, I2, D2, N0 }, 56 bits
	{ OP_DEC_D4, OP_DEC_I2, OP_DEC_I2, OP_DEC_N0 }, // 0x03: { D4, I2, I2, N0 }, 48 bits

	{ OP_DEC_D4, OP_DEC_I4, OP_DEC_N0, OP_DEC_N0 }, // 0x04: { D4, I4, N0, N0 }, 41 bits
	{ OP_DEC_D2, OP_DEC_I2, OP_DEC_D4, OP_DEC_N0 }, // 0x05: { D2, I2, D4, N0 }, 56 bits
	{ OP_DEC_D2, OP_DEC_I2, OP_DEC_D2, OP_DEC_I2 }, // 0x06: { D2, I2, D2, I2 }, 48 bits
	{ OP_DEC_D2, OP_DEC_I2, OP_DEC_I2, OP_DEC_D2 }, // 0x07: { D2, I2, I2, D2 }, 48 bits

	{ OP_DEC_D2, OP_DEC_I2, OP_DEC_I2, OP_DEC_I2 }, // 0x08: { D2, I2, I2, I2 }, 40 bits
	{ OP_DEC_D2, OP_DEC_I2, OP_DEC_I4, OP_DEC_N0 }, // 0x09: { D2, I2, I4, N0 }, 33 bits
	{ OP_DEC_I2, OP_DEC_D2, OP_DEC_D4, OP_DEC_N0 }, // 0x0a: { I2, D2, D4, N0 }, 56 bits
	{ OP_DEC_I2, OP_DEC_D4, OP_DEC_I2, OP_DEC_N0 }, // 0x0b: { I2, D4, I2, N0 }, 48 bits

	{ OP_DEC_I2, OP_DEC_D2, OP_DEC_I2, OP_DEC_D2 }, // 0x0c: { I2, D2, I2, D2 }, 48 bits
	{ OP_DEC_I2, OP_DEC_D2, OP_DEC_I2, OP_DEC_I2 }, // 0x0d: { I2, D2, I2, I2 }, 40 bits
	{ OP_DEC_I2, OP_DEC_D2, OP_DEC_I4, OP_DEC_N0 }, // 0x0e: { I2, D2, I4, N0 }, 33 bits
	{ OP_DEC_I2, OP_DEC_I2, OP_DEC_D4, OP_DEC_N0 }, // 0x0f: { I2, I2, D4, N0 }, 48 bits

	{ OP_DEC_I2, OP_DEC_I2, OP_DEC_D2, OP_DEC_I2 }, // 0x10: { I2, I2, D2, I2 }, 40 bits
	{ OP_DEC_I2, OP_DEC_I2, OP_DEC_I2, OP_DEC_D2 }, // 0x11: { I2, I2, I2, D2 }, 40 bits
	{ OP_DEC_I2, OP_DEC_I2, OP_DEC_I2, OP_DEC_I2 }, // 0x12: { I2, I2, I2, I2 }, 32 bits
	{ OP_DEC_I2, OP_DEC_I2, OP_DEC_I4, OP_DEC_N0 }, // 0x13: { I2, I2, I4, N0 }, 25 bits

	{ OP_DEC_I4, OP_DEC_D4, OP_DEC_N0, OP_DEC_N0 }, // 0x14: { I4, D4, N0, N0 }, 41 bits
	{ OP_DEC_I4, OP_DEC_D2, OP_DEC_I2, OP_DEC_N0 }, // 0x15: { I4, D2, I2, N0 }, 33 bits
	{ OP_DEC_I4, OP_DEC_I2, OP_DEC_D2, OP_DEC_N0 }, // 0x16: { I4, I2, D2, N0 }, 33 bits
	{ OP_DEC_I4, OP_DEC_I2, OP_DEC_I2, OP_DEC_N0 }, // 0x17: { I4, I2, I2, N0 }, 25 bits

	{ OP_DEC_I4, OP_DEC_I4, OP_DEC_N0, OP_DEC_N0 }, // 0x18: { I4, I4, N0, N0 }, 18 bits
	{ OP_DEC_I8, OP_DEC_N0, OP_DEC_N0, OP_DEC_N0 }, // 0x19: { I8, N0, N0, N0 }, 8 bits
};

static inline uint64_t bswap(uint64_t value)
{
	return (uint64_t)((value & (uint64_t)0x00000000000000ff) << 56) |
	       (uint64_t)((value & (uint64_t)0x000000000000ff00) << 40) |
	       (uint64_t)((value & (uint64_t)0x0000000000ff0000) << 24) |
	       (uint64_t)((value & (uint64_t)0x00000000ff000000) << 8) |
	       (uint64_t)((value & (uint64_t)0x000000ff00000000) >> 8) |
	       (uint64_t)((value & (uint64_t)0x0000ff0000000000) >> 24) |
	       (uint64_t)((value & (uint64_t)0x00ff000000000000) >> 40) |
	       (uint64_t)((value & (uint64_t)0xff00000000000000) >> 56);
}

static inline uint64_t read_bits(struct sw842_param_decomp *p, uint32_t n)
{
	uint64_t value = p->buffer >> (WSIZE - n);
	if (n == 0)
		value = 0;

	if (p->bits < n) {
#ifdef ENABLE_ERROR_HANDLING
	if ((p->in - p->istart + 1) * sizeof(uint64_t) > p->ilen) {
		if (p->errorcode == 0)
			p->errorcode = -EINVAL;
		return 0;
	}
#endif
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
		// Avoid shift by 64 (only shifts of strictly less bits bits are allowed by the standard)
		p->buffer = ((p->buffer << (n - p->bits - 1)) << 1);
		p->bits += WSIZE - n;
		p->buffer *= (p->bits > 0);
	} else {
		p->bits -= n;
		p->buffer <<= n;
	}

	return value;
}

static inline uint64_t get_index(struct sw842_param_decomp *p, uint8_t size,
				 uint64_t index, uint64_t fsize)
{
	uint64_t offset;
	uint64_t total = (p->out - p->ostart) * sizeof(uint64_t);

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

#ifdef ENABLE_ERROR_HANDLING
	if (offset + size > total) {
		if (p->errorcode == 0) {
			printf("index%x %lx points past end %lx\n", size,
			       (unsigned long)offset, (unsigned long)total);
			p->errorcode = -EINVAL;
		}
		return 0;
	}
#endif

	return offset;
}

static inline void do_op(struct sw842_param_decomp *p, uint8_t op)
{
#ifdef ENABLE_ERROR_HANDLING
	if (op >= OPS_MAX) {
		p->errorcode = -EINVAL;
		return;
	}
#endif

	uint64_t output_word = 0;
	uint32_t bits = 0;

	for (int i = 0; i < 4; i++) {
		uint64_t value;

		uint32_t dec_template = dec_templates[op][i][0];
		//printf("op is %x\n", dec_template & 0x7F);
		uint32_t is_index = (dec_template >> 7);
		uint32_t dst_size = dec_templates[op][i][1];

		value = read_bits(p, dec_template & 0x7F);
#ifdef ENABLE_ERROR_HANDLING
		if (p->errorcode != 0)
			return;
#endif

		if (is_index) {
			uint64_t offset = get_index(
				p, dst_size, value,
				fifo_sizes[dst_size >> 2]);
#ifdef ENABLE_ERROR_HANDLING
			if (p->errorcode != 0)
				return;
#endif
			offset >>= 1;
			__global uint16_t *ostart16 =
				(__global uint16_t *)p->ostart;
			value = (((uint64_t)ostart16[offset])) |
				(((uint64_t)ostart16[offset + 1]) << 16) |
				(((uint64_t)ostart16[offset + 2]) << 32) |
				(((uint64_t)ostart16[offset + 3]) << 48);
			value &= masks[dst_size >> 2];
			value <<= (WSIZE - (dst_size << 3));
			value = bswap(value);
		}
		output_word |= value
			       << (64 - (dst_size << 3) - bits);
		bits += dst_size << 3;
	}
#ifdef ENABLE_ERROR_HANDLING
	if ((p->out - p->ostart) * sizeof(uint64_t) + 8 > p->olen) {
		p->errorcode = -ENOSPC;
		return;
	}
#endif
	*p->out++ = bswap(output_word);
}

static inline int decompress_core(__global const uint64_t *RESTRICT_UNLESS_INPLACE in, size_t ilen,
				  __global uint64_t *RESTRICT_UNLESS_INPLACE out, size_t *olen)
{
	struct sw842_param_decomp p;
	p.ostart = p.out = out;
	p.in = in;
#ifdef ENABLE_ERROR_HANDLING
	p.istart = p.in;
	p.ilen = ilen;
	p.olen = *olen;
	p.errorcode = 0;
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

	*olen = 0;

	uint64_t op;

	do {
		op = read_bits(&p, OP_BITS);
#ifdef ENABLE_ERROR_HANDLING
		if (p.errorcode != 0)
			return p.errorcode;
#endif

		switch (op) {
		case OP_REPEAT:
			op = read_bits(&p, REPEAT_BITS);
#ifdef ENABLE_ERROR_HANDLING
			if (p.errorcode != 0)
				return p.errorcode;

			if (p.out == out) /* no previous bytes */
				return -EINVAL;
#endif
			// copy op + 1
			op++;

#ifdef ENABLE_ERROR_HANDLING
			if ((p.out - p.ostart) * sizeof(uint64_t) + op * 8 > p.olen)
				return -ENOSPC;
#endif

			while (op-- > 0) {
				*p.out = *(p.out - 1);
				p.out++;
			}
			break;
		case OP_ZEROS:
#ifdef ENABLE_ERROR_HANDLING
			if ((p.out - p.ostart) * sizeof(uint64_t) + 8 > p.olen)
				return -ENOSPC;
#endif
			*p.out = 0;
			p.out++;
			break;
		case OP_END:
			break;
		default:
			do_op(&p, op);
#ifdef ENABLE_ERROR_HANDLING
			if (p.errorcode != 0)
				return p.errorcode;
#endif
		}
	} while (op != OP_END);

	/*
	 * crc(0:31) is saved in compressed data starting with the
	 * next bit after End of stream template.
	 */
#ifndef DISABLE_CRC
	op = read_bits(&p, CRC_BITS);
#ifdef ENABLE_ERROR_HANDLING
	if (p.errorcode != 0)
		return p.errorcode;
#endif

	/*
	 * Validate CRC saved in compressed data.
	 */
	// FIXME: Implement CRC32 for OpenCL
	//if (crc != (uint64_t)crc32_be(0, p.ostart, (p.out - p.ostart) * sizeof(uint64_t))) {
	if (false) {
		return -EINVAL;
	}
#endif

	*olen = (p.out - p.ostart) * sizeof(uint64_t);

	return 0;
}

__kernel void decompress(__global const uint64_t *RESTRICT_UNLESS_INPLACE in,
			 ulong inOffset, __global const ulong *ilen,
			 __global uint64_t *RESTRICT_UNLESS_INPLACE out,
			 ulong outOffset, __global ulong *olen,
			 ulong numChunks, __global const ulong *chunkShuffleMap,
			 __global int *returnValues)
{
	size_t chunk_num = get_global_id(0);
	if (chunk_num >= numChunks)
		return;

	if (chunkShuffleMap != NULL)
		chunk_num = chunkShuffleMap[chunk_num];

	__global uint64_t *my_out = out + (outOffset / 8) + ((CL842_CHUNK_SIZE / 8) * chunk_num);
	__global const uint64_t *my_in = in + (inOffset / 8) + ((CL842_CHUNK_STRIDE / 8) * chunk_num);

#if defined(USE_MAYBE_COMPRESSED_CHUNKS) || defined(USE_INPLACE_COMPRESSED_CHUNKS)
	if (my_in[0] != ((__constant const uint64_t *)LIB842_COMPRESSED_CHUNK_MARKER)[0] ||
	    my_in[1] != ((__constant const uint64_t *)LIB842_COMPRESSED_CHUNK_MARKER)[1]) {
#ifdef USE_MAYBE_COMPRESSED_CHUNKS
		// Copy uncompressed chunk from temporary input buffer to output buffer
		for (size_t i = 0; i < CL842_CHUNK_SIZE / 8; i++) {
			my_out[i] = my_in[i];
		}
#endif
		if (olen)
			olen[chunk_num] = CL842_CHUNK_SIZE;
		if (returnValues)
			returnValues[chunk_num] = 0;
		return;
	}

	// Read compressed chunk size and skip to the beginning of the chunk
	// (the end of the chunk matches the end of the input chunk buffer)
	my_in += (CL842_CHUNK_SIZE - my_in[2]) / 8;
#endif

	size_t my_ilen = ilen != NULL ? ilen[chunk_num] : (size_t)-1;
	size_t my_olen = olen != NULL ? olen[chunk_num] : (size_t)-1;

	int ret = decompress_core(my_in, my_ilen, my_out, &my_olen);
	if (olen)
		olen[chunk_num] = my_olen;
	if (returnValues)
		returnValues[chunk_num] = ret;
}
