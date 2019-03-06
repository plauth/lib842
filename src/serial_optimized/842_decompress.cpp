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
#include "kerneldeps.h"

/* rolling fifo sizes */
#define I2_FIFO_SIZE	(2 * (1 << I2_BITS))
#define I4_FIFO_SIZE	(4 * (1 << I4_BITS))
#define I8_FIFO_SIZE	(8 * (1 << I8_BITS))

static uint8_t decomp_ops[OPS_MAX][4] = {
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

static int do_data(struct sw842_param_decomp *p, uint8_t n)
{
	uint64_t v = stream_read_bits(p->stream, n*8);

	switch (n) {
	case 2:
		write16(p->out, swap_be_to_native16(v));
		break;
	case 4:
		write32(p->out, swap_be_to_native32(v));
		break;
	case 8:
		write64(p->out, swap_be_to_native64(v));
		break;
	default:
		return -EINVAL;
	}

	p->out += n;
	p->olen -= n;

	return 0;
}

static int __do_index(struct sw842_param_decomp *p, uint8_t size, uint8_t bits, uint64_t fsize)
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

	if (offset + size > total) {
		fprintf(stderr, "index%x %lx points past end %lx\n", size,
			 (unsigned long)offset, (unsigned long)total);
		return -EINVAL;
	}

	memcpy(p->out, &p->ostart[offset], size);
	p->out += size;
	p->olen -= size;

	return 0;
}

static int do_index(struct sw842_param_decomp *p, uint8_t n)
{
	switch (n) {
	case 2:
		return __do_index(p, 2, I2_BITS, I2_FIFO_SIZE);
	case 4:
		return __do_index(p, 4, I4_BITS, I4_FIFO_SIZE);
	case 8:
		return __do_index(p, 8, I8_BITS, I8_FIFO_SIZE);
	default:
		return -EINVAL;
	}
}

static int do_op(struct sw842_param_decomp *p, uint8_t o)
{
	for (int i = 0; i < 4; i++) {
		uint8_t op = decomp_ops[o][i];


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
			fprintf(stderr, "Internal error, invalid op %x\n", op);
			exit(-EINVAL);
		}
	}
	return 0;
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
		printf("template is %lx\n", (unsigned long)op);
		#endif

		switch (op) {
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
		default: /* use template */
			do_op(p, op);
			break;
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


