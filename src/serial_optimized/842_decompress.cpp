/*
 * 842 Software Decompression
 *
 * Copyright (C) 2015 Dan Streetman, IBM Corp
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * See 842.h for details of the 842 compressed format.
 */

#include "842-internal.h"

/* rolling fifo sizes */
#define I2_FIFO_SIZE	(2 * (1 << I2_BITS))
#define I4_FIFO_SIZE	(4 * (1 << I4_BITS))
#define I8_FIFO_SIZE	(8 * (1 << I8_BITS))

#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define round_down(x, y) ((x) & ~__round_mask(x, y))

template<uint8_t N> static inline void do_data(struct sw842_param_decomp *p)
{
	switch (N) {
	case 2:
		write16(p->out, swap_be_to_native16(stream_read_bits(p->stream, 16)));
		break;
	case 4:
		write32(p->out, swap_be_to_native32(stream_read_bits(p->stream, 32)));
		break;
	case 8:
		write64(p->out, swap_be_to_native64(stream_read_bits(p->stream, 64)));
		break;
	}

	p->out += N;
	p->olen -= N;
}

static inline void do_index(struct sw842_param_decomp *p, uint8_t size, uint8_t bits, uint64_t fsize)
{
	uint64_t index, offset, total = round_down(p->out - p->ostart, 8);

	index = stream_read_bits(p->stream, bits);

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
	p->olen -= size;
}

/**
 * sw842_decompress
 *
 * Decompress the 842-compressed buffer of length @ilen at @in
 * to the output buffer @out, using no more than @olen bytes.
 *
 * The compressed buffer must be only a single 842-compressed buffer,
 * with the standard format described in the comments in 842.h
 * Processing will stop when the 842 "END" template is detected,
 * not the end of the buffer.
 *
 * Returns: 0 on success, error on failure.  The @olen parameter
 * will contain the number of output bytes written on success, or
 * 0 on error.
 */
int sw842_decompress(const uint8_t *in, unsigned int ilen,
		     uint8_t *out, unsigned int *olen)
{
	struct sw842_param_decomp *p = (struct sw842_param_decomp *) malloc(sizeof(struct sw842_param_decomp)); 
	uint64_t op, rep, total;

	p->in = (uint8_t *)in;
	p->ilen = ilen;
	p->out = out;
	p->ostart = out;
	p->olen = *olen;

	p->stream = stream_open((uint8_t *)in, ilen);

	total = p->olen;

	*olen = 0;

	do {
		op = stream_read_bits(p->stream, OP_BITS);

		#ifdef DEBUG
		printf("template is %llx\n", op);
		#endif

		switch (op) {
			case 0x00: 	// { D8, N0, N0, N0 }, 64 bits
	        	do_data<8>(p);
	    	    break;
	        case 0x01:	// { D4, D2, I2, N0 }, 56 bits
	        	do_data<4>(p);
	        	do_data<2>(p);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
	        case 0x02:	// { D4, I2, D2, N0 }, 56 bits
	        	do_data<4>(p);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_data<2>(p);
	    	    break;
			case 0x03: 	// { D4, I2, I2, N0 }, 48 bits
	        	do_data<4>(p);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x04:	// { D4, I4, N0, N0 }, 41 bits
	        	do_data<4>(p);
	        	do_index(p, 4, I4_BITS, I4_FIFO_SIZE);
	    	    break;
			case 0x05:	// { D2, I2, D4, N0 }, 56 bits
				do_data<2>(p);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_data<4>(p);
	    	    break;
			case 0x06:	// { D2, I2, D2, I2 }, 48 bits
				do_data<2>(p);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(p);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x07:	// { D2, I2, I2, D2 }, 48 bits
				do_data<2>(p);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(p);
	    	    break;
			case 0x08:	// { D2, I2, I2, I2 }, 40 bits
				do_data<2>(p);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x09:	// { D2, I2, I4, N0 }, 33 bits
				do_data<2>(p);
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_index(p, 4, I4_BITS, I4_FIFO_SIZE);
	    	    break;
			case 0x0a:	// { I2, D2, D4, N0 }, 56 bits
	        	do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	        	do_data<2>(p);
	        	do_data<4>(p);
	    	    break;
			case 0x0b:	// { I2, D4, I2, N0 }, 48 bits
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<4>(p);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x0c:	// { I2, D2, I2, D2 }, 48 bits
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(p);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(p);
	    	    break;
			case 0x0d:	// { I2, D2, I2, I2 }, 40 bits
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(p);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x0e:	// { I2, D2, I4, N0 }, 33 bits
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(p);
				do_index(p, 4, I4_BITS, I4_FIFO_SIZE);
	    	    break;
			case 0x0f:	// { I2, I2, D4, N0 }, 48 bits
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<4>(p);
	    	    break;
			case 0x10:	// { I2, I2, D2, I2 }, 40 bits
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(p);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x11:	// { I2, I2, I2, D2 }, 40 bits
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(p);
	    	    break;
			case 0x12:	// { I2, I2, I2, I2 }, 32 bits
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x13:	// { I2, I2, I4, N0 }, 25 bits
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(p, 4, I4_BITS, I4_FIFO_SIZE);
	    	    break;
			case 0x14:	// { I4, D4, N0, N0 }, 41 bits
				do_index(p, 4, I4_BITS, I4_FIFO_SIZE);
				do_data<4>(p);
	    	    break;
			case 0x15:	// { I4, D2, I2, N0 }, 33 bits
				do_index(p, 4, I4_BITS, I4_FIFO_SIZE);
				do_data<2>(p);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x16:	// { I4, I2, D2, N0 }, 33 bits
				do_index(p, 4, I4_BITS, I4_FIFO_SIZE);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_data<2>(p);
	    	    break;
			case 0x17:	// { I4, I2, I2, N0 }, 25 bits
				do_index(p, 4, I4_BITS, I4_FIFO_SIZE);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
				do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	    	    break;
			case 0x18:	// { I4, I4, N0, N0 }, 18 bits
				do_index(p, 4, I4_BITS, I4_FIFO_SIZE);
				do_index(p, 4, I4_BITS, I4_FIFO_SIZE);
	    	    break;
			case 0x19:	// { I8, N0, N0, N0 }, 8 bits
				do_index(p, 8, I8_BITS, I8_FIFO_SIZE);
	    	    break;
	    	case OP_REPEAT:
				rep = stream_read_bits(p->stream, REPEAT_BITS);
				/* copy rep + 1 */
				rep++;

				while (rep-- > 0) {
					memcpy(p->out, p->out - 8, 8);
					p->out += 8;
					p->olen -= 8;
				}
				break;
			case OP_ZEROS:
				memset(p->out, 0, 8);
				p->out += 8;
				p->olen -= 8;
				break;
			case OP_END:
				break;
	        default:
	        	fprintf(stderr, "Invalid op template: %llx\n", op);
	        	return -EINVAL;
	    }
	} while (op != OP_END);

	/*
	 * crc(0:31) is saved in compressed data starting with the
	 * next bit after End of stream template.
	 */
	#ifndef DISABLE_CRC
	uint64_t crc = swap_be_to_native32(stream_read_bits(p->stream, CRC_BITS));
	
	/*
	 * Validate CRC saved in compressed data.
	 */
	if (crc != (uint64_t) crc32_be(0, out, total - p->olen)) {
		fprintf(stderr, "CRC mismatch for decompression\n");
		return -EINVAL;
	}
	#endif

	*olen = total - p->olen;

	stream_close(p->stream);
	free(p);

	return 0;
}


