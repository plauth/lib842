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

static inline uint64_t swap_be_to_native64(uint64_t value)
{
#ifdef __ENDIAN_LITTLE__
	return as_ulong(as_uchar8(value).s76543210);
#else
	return value;
#endif
}

static inline uint64_t swap_native_to_be64(uint64_t value)
{
#ifdef __ENDIAN_LITTLE__
	return as_ulong(as_uchar8(value).s76543210);
#else
	return value;
#endif
}

static inline uint16_t swap_be_to_native16(uint16_t value)
{
#ifdef __ENDIAN_LITTLE__
	return as_ushort(as_uchar2(value).s10);
#else
	return value;
#endif
}

static inline uint16_t swap_native_to_be16(uint16_t value)
{
#ifdef __ENDIAN_LITTLE__
	return as_ushort(as_uchar2(value).s10);
#else
	return value;
#endif
}

static inline uint64_t read_bits(struct sw842_param_decomp *p, uint32_t n)
{
	uint64_t value = p->buffer >> (WSIZE - n);
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
		p->lookAheadBuffer[5] = swap_be_to_native64(*p->in);
#else
		p->buffer = swap_be_to_native64(*p->in);
#endif
		p->in++;
		value |= p->buffer >> (WSIZE - (n - p->bits));
		// Avoid shift by 64 (only shifts of strictly less bits are allowed by the standard)
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
		printf("index%x %lx points past end %lx\n", size,
		       (unsigned long)offset, (unsigned long)total);
		p->errorcode = -EINVAL;
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

	// TODOXXX explain the patterns those formulas are based on
	uint8_t opbits = 64 - ((op % 5) + 1) / 2 * 8 - ((op % 5) / 4) * 7
			    - ((op / 5) + 1) / 2 * 8 - ((op / 5) / 4) * 7;
	uint64_t params = read_bits(p, opbits);
#ifdef ENABLE_ERROR_HANDLING
	if (p->errorcode != 0)
		return;
#endif

	for (int i = 0; i < 4; i++) {
		// TODOXXX explain the patterns those formulas are based on
		uint8_t opchunk = (i < 2) ? op / 5 : op % 5;
		uint32_t is_index = (i & 1) * (opchunk & 1) + ((i & 1) ^ 1) * (opchunk >= 2);
		uint32_t dst_size = 2 + (opchunk >= 4) * (1 - 2 * (i % 2)) * 2;
		uint8_t num_bits = (i & 1) * (16 - (opchunk % 2) * 8 - (opchunk >= 4) * 16) +
				   ((i & 1) ^ 1) * (16 - (opchunk / 2) * 8 + (opchunk >= 4) * 9);

		// https://stackoverflow.com/a/28703383
		uint64_t bitsmask = ((uint64_t)-(num_bits != 0)) &
				    (((uint64_t)-1) >> (64 - num_bits));
		uint64_t value = (params >> (opbits - num_bits)) & bitsmask;
		opbits -= num_bits;

		if (is_index) {
			// TODOXXX explain how this relates to In_FIFO_SIZE constants
			uint64_t offset = get_index(
				p, dst_size, value,
				2048 - 1536 * ((dst_size >> 2) < 1));
#ifdef ENABLE_ERROR_HANDLING
			if (p->errorcode != 0)
				return;
#endif
			offset >>= 1;
			__global uint16_t *ostart16 =
				(__global uint16_t *)p->ostart;
			value = (((uint64_t)swap_be_to_native16(ostart16[offset])) << 48) |
				(((uint64_t)swap_be_to_native16(ostart16[offset + 1])) << 32) |
				(((uint64_t)swap_be_to_native16(ostart16[offset + 2])) << 16) |
				(((uint64_t)swap_be_to_native16(ostart16[offset + 3])));
			value >>= (WSIZE - (dst_size << 3));
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
	*p->out++ = swap_native_to_be64(output_word);
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
	p.lookAheadBuffer[0] = swap_be_to_native64(*p.in++);
	p.lookAheadBuffer[1] = swap_be_to_native64(*p.in++);
	p.lookAheadBuffer[2] = swap_be_to_native64(*p.in++);
	p.lookAheadBuffer[3] = swap_be_to_native64(*p.in++);
	p.lookAheadBuffer[4] = swap_be_to_native64(*p.in++);
	p.lookAheadBuffer[5] = swap_be_to_native64(*p.in++);
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
		case (OPS_MAX - 1): {
			// The I8 opcode doesn't fit into the same patterns
			// as the first 25 opcodes, so it's handled separately
			uint64_t value = read_bits(&p, 8);
#ifdef ENABLE_ERROR_HANDLING
			if (p.errorcode != 0)
				return p.errorcode;
#endif

			uint64_t offset = get_index(
				&p, 8, value, I8_FIFO_SIZE);
#ifdef ENABLE_ERROR_HANDLING
			if (p.errorcode != 0)
				return p.errorcode;
			if ((p.out - p.ostart) * sizeof(uint64_t) + 8 > p.olen)
				return -ENOSPC;
#endif
			*p.out++ = p.ostart[offset >> 3];
		}
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
