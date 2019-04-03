#include "842-internal.h"

/* rolling fifo sizes */
#define I2_FIFO_SIZE	(2 * (1 << I2_BITS))
#define I4_FIFO_SIZE	(4 * (1 << I4_BITS))
#define I8_FIFO_SIZE	(8 * (1 << I8_BITS))

#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define round_down(x, y) ((x) & ~__round_mask(x, y))

/* number of bits in a buffered word */
#define WSIZE 64 //sizeof(uint64_t)

#define CHUNK_SIZE 1024

#ifdef STRICT
__constant__ uint16_t fifo_sizes[9] = {
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
#endif


__constant__ uint8_t dec_templates[26][4][2] = { // params size in bits
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

__device__ inline uint64_t bswap(uint64_t value)
{
	asm("{\n\t"
		"		.reg .b32 %li,%lo,%hi,%ho;\n\t"
		"		mov.b64 {%li,%hi}, %0;\n\t"
		"		prmt.b32 %lo, %li, %hi, 0x4567;\n\t"
		" 		prmt.b32 %ho, %li, %hi, 0x0123;\n\t"
		"		mov.b64 %0, {%lo,%ho};\n\t"
		"}"
		: "+l"(value)
	);
	return value;
}


__device__ inline uint64_t read_bits(struct sw842_param_decomp *p, uint32_t num_chunks, uint32_t n)
{
  uint64_t value = p->buffer >> (WSIZE - n);
  //value &= (n > 0) ? 0xFFFFFFFFFFFFFFFF : 0x0000000000000000;
  asm("{\n\t"
	"		.reg .pred %p;\n\t"
	"		setp.ls.u32 %p, %1, 0;\n\t"
	"@%p	mov.u64 %0, 0;\n\t"
	"}"
	: "+l"(value) : "r"(n)
  );

  /*
  asm("{\n\t"
	"		.reg .pred %p, %q;\n\t"
	"		.reg .b32 %li,%lo,%hi,%ho;\n\t"
	"		.reg .b64 %tmp;\n\t"
	"		.reg .u32 %n;\n\t"
	// value = (n > 0) ? value : 0
	"		setp.hi.u32 %p, %5, 0;\n\t"		// n > 0?
	"		selp.u64 %0, %0, 0, %p;\n\t"	// set value to value if predicate is true, otherwise to 0
	// if (p->bits < n)
	"		setp.lo.u32 %p, %3, %5;\n\t"
	// begin: p->buffer = read_word(p)
	// swap_be_to_native64(p->in)
	"@%p	mov.b64 {%li,%hi}, %4;\n\t"
	"@%p	prmt.b32 %lo, %li, %hi, 0x4567;\n\t"
	"@%p	prmt.b32 %ho, %li, %hi, 0x0123;\n\t"
	"@%p	mov.b64 %1, {%lo,%ho};\n\t"
	// p->in++
	"@%p	add.u64 %2, %2, 8;\n\t"
	// end: p->buffer = read_word(p)

	// n = WSIZE - (n - p->bits)
	"@%p	sub.u32 %n, %5, %3;\n\t"
	"@%p	sub.u32 %n, 0x40, %n;\n\t"
	// value |= p->buffer >> n
	"		shr.b64 %tmp, %1, %n;\n\t"
	"@%p	or.b64 %0, %0, %tmp;\n\t"
	// p->buffer <<= n - p->bits
	"		sub.u32 %n, %5, %3;\n\t"
	"@%p	shl.b64 %1, %1, %n;\n\t"
	// p->bits += WSIZE - n
	"@%p	add.u32 %3, %3, 0x40;\n\t"
	"@%p	sub.u32 %3, %3, %5;\n\t"
	// p->buffer *= (p->bits > 0)
	"@%p	setp.hi.u32 %q, %3, 0;\n\t"
	"@%p	selp.u64 %1, %1, 0, %q;\n\t"
	// else
    // p->bits -=n; p->buffer <<=n;
	"@!%p	sub.u32 %3, %3, %5;\n\t"
	"@!%p	shl.b64 %1, %1, %5;\n\t"
	"}"
	: "+l"(value), "+l"(p->buffer), "+l"(p->in), "+r"(p->bits) : "l"(*p->in), "r"(n)
  );
  */

  if (p->bits < n) {
    p->buffer = bswap(*p->in);
    p->in += num_chunks;
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

#ifdef STRICT
__device__ inline uint64_t get_index(struct sw842_param_decomp *p, uint8_t size, uint64_t index, uint64_t fsize)
{
	uint64_t offset;
	uint64_t total = round_down(((uint8_t*)p->out) - ((const uint8_t *)p->ostart), 8);

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
#endif

__global__ void cuda842_decompress(uint64_t *in, uint32_t ilen, uint64_t *out, uint32_t num_chunks)
{
	unsigned int chunk_num = blockIdx.x * blockDim.x + threadIdx.x;

	struct sw842_param_decomp p;
	p.ostart = p.out = out + ((CHUNK_SIZE / 8) * chunk_num);
  	//p.in = (in + ((CHUNK_SIZE / 8 * 2) * chunk_num));
  	p.in = in + chunk_num;

	p.buffer = 0;
	p.bits = 0;

	uint64_t op;

	uint64_t output_word;
	uint8_t bits;

	do {
		op = read_bits(&p, num_chunks, OP_BITS);

		output_word = 0;
		bits = 0;

		switch (op) {
	    	case OP_REPEAT:
				op = read_bits(&p, num_chunks, REPEAT_BITS);
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
					uint32_t num_bits = dec_template & 0x7F;
					uint32_t dst_size = dec_templates[op][i][1];

					value = read_bits(&p, num_chunks, num_bits);

					if(is_index) {
#ifdef STRICT
						uint64_t offset = get_index(&p, dst_size, value, fifo_sizes[dst_size]);
#else
						//uint32_t offset = value * dst_size;
#endif

						asm("{\n\t"
						"		.reg .pred %pr4, %pr8;\n\t"
						"		.reg .u16 %val16_0, %val16_1, %val16_2, %val16_3;\n\t"
						"		.reg .u32 %val32;\n\t"
						"		.reg .u64 %addr, %result;\n\t"

						"		setp.hi.u32 %pr4, %2, 2;\n\t"
						"		setp.eq.u32 %pr8, %2, 8;\n\t"
#ifdef STRICT
						"		add.u64 %addr, %1, %3;\n\t"
#else
						"		cvt.u64.u32 %addr, %2;\n\t"
						"		mul.lo.u64 %addr, %0, %addr;\n\t"
						"		add.u64 %addr, %addr, %1;\n\t"
#endif

						"		ld.global.b16 %val16_0, [%addr];\n\t"
						"@%pr4	ld.global.b16 %val16_1, [%addr+2];\n\t"
						"@%pr8	ld.global.b16 %val16_2, [%addr+4];\n\t"
						"@%pr8	ld.global.b16 %val16_3, [%addr+6];\n\t"
						"		cvt.u64.u16 %result, %val16_0;\n\t"
						"@%pr4	mov.b32 %val32, {%val16_0, %val16_1};\n\t"
						"@%pr4	cvt.u64.u32 %result, %val32;\n\t"
						"@%pr8	mov.b64 %result, {%val16_0, %val16_1, %val16_2, %val16_3};\n\t"
						"		shl.b32 %val32, %2, 3;\n\t"
						"		sub.u32 %val32, 64, %val32;\n\t"
						"		shl.b64 %result, %result, %val32;\n\t"

						"		.reg .b32 %li,%lo,%hi,%ho;\n\t"
						"		mov.b64 {%li,%hi}, %result;\n\t"
						"		prmt.b32 %lo, %li, %hi, 0x4567;\n\t"
						" 		prmt.b32 %ho, %li, %hi, 0x0123;\n\t"
						"		mov.b64 %result, {%lo,%ho};\n\t"
						"		mov.b64 %0, %result;\n\t"

						"}"
#ifdef STRICT
						: "+l"(value) : "l"(p.ostart), "r"(dst_size), "l"(offset)
#else
						: "+l"(value) : "l"(p.ostart), "r"(dst_size)
#endif

						 );
					}

					output_word |= value << (64 - (dst_size<<3) - bits);
					bits += dst_size<<3;
				}
				*p.out++ = bswap(output_word);
	    }
	} while (op != OP_END);

	return;
}
