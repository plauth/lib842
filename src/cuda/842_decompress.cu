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

/* read a single uint64_t from memory */
__device__ static inline uint64_t stream_read_word(struct sw842_param_decomp *p)
{
  uint64_t w = swap_be_to_native64(*p->in++);
  return w;
}

/* read 0 <= n <= 64 bits */
__device__ static inline uint64_t read_bits(struct sw842_param_decomp *p, uint8_t n)
{
  uint64_t value = p->buffer >> (WSIZE - n);

  if (p->bits < n) {
    /* fetch WSIZE bits  */
    p->buffer = stream_read_word(p);
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

template<uint8_t N> __device__ static inline void do_data(struct sw842_param_decomp *p)
{
	switch (N) {
	case 2:
		write16(p->out, swap_be_to_native16(read_bits(p, 16)));
		break;
	case 4:
		write32(p->out, swap_be_to_native32(read_bits(p, 32)));
		break;
	case 8:
		write64(p->out, swap_be_to_native64(read_bits(p, 64)));
		break;
	}

	p->out += N;
}

__device__ static inline void do_index(struct sw842_param_decomp *p, uint8_t size, uint8_t bits, uint64_t fsize)
{
	uint64_t index, offset, total = round_down(p->out - p->ostart, 8);

	index = read_bits(p, bits);

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

	memcpy(p->out, &p->ostart[offset], size);
	p->out += size;
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
	do {
		op = read_bits(&p, OP_BITS);

		#ifdef DEBUG
		printf("template is %llx\n", op);
		#endif

		switch (op) {
			case 0x00: 	// { D8, N0, N0, N0 }, 64 bits
	        	do_data<8>(&p);
	    	    break;
	        case 0x01:	// { D4, D2, I2, N0 }, 56 bits
	        	do_data<4>(&p);
	        	do_data<2>(&p);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
	        case 0x02:	// { D4, I2, D2, N0 }, 56 bits
	        	do_data<4>(&p);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_data<2>(&p);
	    	    break;
			case 0x03: 	// { D4, I2, I2, N0 }, 48 bits
	        	do_data<4>(&p);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x04:	// { D4, I4, N0, N0 }, 41 bits
	        	do_data<4>(&p);
	        	do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
	    	    break;
			case 0x05:	// { D2, I2, D4, N0 }, 56 bits
				do_data<2>(&p);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_data<4>(&p);
	    	    break;
			case 0x06:	// { D2, I2, D2, I2 }, 48 bits
				do_data<2>(&p);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(&p);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x07:	// { D2, I2, I2, D2 }, 48 bits
				do_data<2>(&p);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(&p);
	    	    break;
			case 0x08:	// { D2, I2, I2, I2 }, 40 bits
				do_data<2>(&p);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x09:	// { D2, I2, I4, N0 }, 33 bits
				do_data<2>(&p);
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
	    	    break;
			case 0x0a:	// { I2, D2, D4, N0 }, 56 bits
	        	do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_data<2>(&p);
	        	do_data<4>(&p);
	    	    break;
			case 0x0b:	// { I2, D4, I2, N0 }, 48 bits
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<4>(&p);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x0c:	// { I2, D2, I2, D2 }, 48 bits
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(&p);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(&p);
	    	    break;
			case 0x0d:	// { I2, D2, I2, I2 }, 40 bits
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(&p);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x0e:	// { I2, D2, I4, N0 }, 33 bits
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(&p);
				do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
	    	    break;
			case 0x0f:	// { I2, I2, D4, N0 }, 48 bits
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<4>(&p);
	    	    break;
			case 0x10:	// { I2, I2, D2, I2 }, 40 bits
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(&p);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x11:	// { I2, I2, I2, D2 }, 40 bits
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(&p);
	    	    break;
			case 0x12:	// { I2, I2, I2, I2 }, 32 bits
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x13:	// { I2, I2, I4, N0 }, 25 bits
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
	    	    break;
			case 0x14:	// { I4, D4, N0, N0 }, 41 bits
				do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
				do_data<4>(&p);
	    	    break;
			case 0x15:	// { I4, D2, I2, N0 }, 33 bits
				do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
				do_data<2>(&p);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x16:	// { I4, I2, D2, N0 }, 33 bits
				do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(&p);
	    	    break;
			case 0x17:	// { I4, I2, I2, N0 }, 25 bits
				do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x18:	// { I4, I4, N0, N0 }, 18 bits
				do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
				do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
	    	    break;
			case 0x19:	// { I8, N0, N0, N0 }, 8 bits
				do_index(&p, 8, I8_BITS, I8_FIFO_SIZE);
	    	    break;
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
	        	printf("Invalid op template: %llx\n", op);
	        	return;
	    }
	} while (op != OP_END);

	return;
}
