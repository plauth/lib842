#include "842-internal.h"

/* rolling fifo sizes */
#define I2_FIFO_SIZE	(2 * (1 << I2_BITS))
#define I4_FIFO_SIZE	(4 * (1 << I4_BITS))
#define I8_FIFO_SIZE	(8 * (1 << I8_BITS))

#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define round_down(x, y) ((x) & ~__round_mask(x, y))

/* number of bits in a buffered word */
#define WSIZE 64 //sizeof(uint64_t)

#define CHUNK_SIZE 4096

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

/* read a single uint64_t from memory */
__device__ static inline uint64_t read_word(struct sw842_param_decomp *p)
{
  uint64_t w;
  asm("{\n\t"
	"	.reg .b32 %li,%lo,%hi,%ho;\n\t"
	// swap_be_to_native64(p->in)
	"	mov.b64 {%li,%hi}, %2;\n\t"
	"	prmt.b32 %lo, %li, %hi, 0x4567;\n\t"
	"	prmt.b32 %ho, %li, %hi, 0x0123;\n\t"
	"	mov.b64 %0, {%lo,%ho};\n\t"
	// p->in++
	"	add.u64 %1, %1, 8;\n\t"
	"}"
	: "=l"(w), "+l"(p->in) : "l"(*p->in)
  );
  return w;
}

__device__ inline uint64_t read_bits(struct sw842_param_decomp *p, uint32_t n)
{
  uint64_t value = p->buffer >> (WSIZE - n);

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
  );*/

  if (p->bits < n) {
    /* fetch WSIZE bits  */
    p->buffer = read_word(p);
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

/* read 0 <= n <= 64 bits
__device__ static inline uint64_t read_bits(struct sw842_param_decomp *p, uint8_t n)
{
  uint64_t value = p->buffer >> (WSIZE - n);
  value &= (n > 0) * 0xFFFFFFFFFFFFFFFF;

  uint8_t additional_bits_required = p->bits < n;
  uint8_t no_additional_bits_required = 1 - additional_bits_required;

  uint64_t new_buffer = read_word(p, additional_bits_required);
  p->buffer = new_buffer * additional_bits_required | p->buffer * no_additional_bits_required;

  value |= (p->buffer >> (WSIZE - (n - p->bits))) * additional_bits_required;
  p->buffer <<= n - (p->bits * additional_bits_required);
  p->bits += (WSIZE - n) * additional_bits_required;
  p->buffer *= (p->bits > 0 || no_additional_bits_required);
  p->bits -= n * no_additional_bits_required;

  return value;
}*/

__device__ inline uint64_t get_index(struct sw842_param_decomp *p, uint8_t size, uint64_t index, uint64_t fsize)
{
	uint64_t offset, total = round_down(p->out - p->ostart, 8);

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

__global__ void cuda842_decompress(const uint8_t *in, unsigned int ilen, uint8_t *out)
{
	unsigned int chunk_num = blockIdx.x * blockDim.x + threadIdx.x;

	struct sw842_param_decomp p;
	p.out = out + (CHUNK_SIZE * chunk_num);
	p.ostart = out + (CHUNK_SIZE * chunk_num);
  	p.in = (uint64_t*) (in + ((CHUNK_SIZE * 2) * chunk_num));
	p.istart = (const uint64_t*) (in + ((CHUNK_SIZE * 2) * chunk_num));
	p.buffer = 0;
	p.bits = 0;

	uint64_t op, rep;

	uint64_t output_word;
	uint64_t values[8];
	uint8_t bits;

	do {
		op = read_bits(&p, OP_BITS);

		#ifdef DEBUG
		printf("template is %llx\n", op);
		#endif

		output_word = 0;
		bits = 0;
		memset(values, 0, 64);

		switch (op) {
	    	case OP_REPEAT:
				rep = read_bits(&p, REPEAT_BITS);
				/* copy rep + 1 */
				rep++;

				while (rep-- > 0) {
					memcpy(p.out, p.out - 8, 8);
					p.out += 8;
				}
				break;
			case OP_ZEROS:
				memset(p.out, 0, 8);
				p.out += 8;
				break;
			case OP_END:
				break;
	        default:
				for(int i = 0; i < 4; i++) {
					// 0-initialize all values-fields
					values[i] = 0;
					values[4 + i] = 0;

					uint8_t dec_template = dec_templates[op][i][0];
					uint8_t is_index = (dec_template >> 7);
					uint8_t num_bits = dec_template & 0x7F;
					uint8_t dst_size = dec_templates[op][i][1];

					values[(4 * is_index) + i] = read_bits(&p, num_bits);

					uint64_t offset = get_index(&p, dst_size, values[4 + i], fifo_sizes[dst_size]);
					memcpy(&values[4 + i], &p.ostart[offset], dst_size);
					values[4 + i] = values[4 + i] << (WSIZE - (dst_size << 3));
					asm("{\n\t"
						"		.reg .b32 %li,%lo,%hi,%ho;\n\t"
						"		mov.b64 {%li,%hi}, %0;\n\t"
						"		prmt.b32 %lo, %li, %hi, 0x4567;\n\t"
						" 		prmt.b32 %ho, %li, %hi, 0x0123;\n\t"
						"		mov.b64 %0, {%lo,%ho};\n\t"
						"}"
						: "+l"(values[4 + i])
					);

					values[i] = values[4 + i] * is_index | values[i];
					output_word |= values[i] << (64 - (dst_size<<3) - bits);
					bits += dst_size<<3;
				}

				asm("{\n\t"
					"		.reg .b32 %li,%lo,%hi,%ho;\n\t"
					// p.out = swap_native_to_be64(p->in)
					"		mov.b64 {%li,%hi}, %2;\n\t"
					"		prmt.b32 %lo, %li, %hi, 0x4567;\n\t"
					" 		prmt.b32 %ho, %li, %hi, 0x0123;\n\t"
					"		mov.b64 %0, {%lo,%ho};\n\t"
					// p.out += 8
					"		add.u64 %1, %1, 8;\n\t"
					"}"
					: "=l"(*((uint64_t*)p.out)), "+l"(p.out) : "l"(output_word)
				);
	    }
	} while (op != OP_END);

	return;
}
