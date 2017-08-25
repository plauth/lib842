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

#include "842.h"

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



#define beN_to_cpu(d, s)					\
	((s) == 2 ? swap_endianness16(get_unaligned16((__be16 *)d)) :	\
	 (s) == 4 ? swap_endianness32(get_unaligned32((__be32 *)d)) :	\
	 (s) == 8 ? swap_endianness64(get_unaligned64((__be64 *)d)) :	\
	 0)

static int next_bits(struct sw842_param_decomp *p, uint64_t *d, uint8_t n);

static int __split_next_bits(struct sw842_param_decomp *p, uint64_t *d, uint8_t n, uint8_t s)
{
	uint64_t tmp = 0;
	int ret;

	if (n <= s) {
		fprintf(stderr, "split_next_bits invalid n %u s %u\n", n, s);
		return -EINVAL;
	}

	ret = next_bits(p, &tmp, n - s);
	if (ret)
		return ret;
	ret = next_bits(p, d, s);
	if (ret)
		return ret;
	*d |= tmp << s;
	return 0;
}

static int next_bits(struct sw842_param_decomp *p, uint64_t *d, uint8_t n)
{
	uint8_t *in = p->in, b = p->bit, bits = b + n;

	if (n > 64) {
		fprintf(stderr, "next_bits invalid n %u\n", n);
		return -EINVAL;
	}

	/* split this up if reading > 8 bytes, or if we're at the end of
	 * the input buffer and would read past the end
	 */
	if (bits > 64)
		return __split_next_bits(p, d, n, 32);
	else if (p->ilen < 8 && bits > 32 && bits <= 56)
		return __split_next_bits(p, d, n, 16);
	else if (p->ilen < 4 && bits > 16 && bits <= 24)
		return __split_next_bits(p, d, n, 8);

	if (DIV_ROUND_UP(bits, 8) > p->ilen)
		return -EOVERFLOW;

	if (bits <= 8)
		*d = *in >> (8 - bits);
	else if (bits <= 16)
		*d = swap_endianness16(get_unaligned16((__be16 *)in)) >> (16 - bits);
	else if (bits <= 32)
		*d = swap_endianness32(get_unaligned32((__be32 *)in)) >> (32 - bits);
	else
		*d = swap_endianness64(get_unaligned64((__be64 *)in)) >> (64 - bits);

	*d &= GENMASK_ULL(n - 1, 0);

	p->bit += n;

	if (p->bit > 7) {
		p->in += p->bit / 8;
		p->ilen -= p->bit / 8;
		p->bit %= 8;
	}

	return 0;
}

static int do_data(struct sw842_param_decomp *p, uint8_t n)
{
	uint64_t v;
	int ret;

	if (n > p->olen)
		return -ENOSPC;

	ret = next_bits(p, &v, n * 8);
	if (ret)
		return ret;

	switch (n) {
	case 2:
		put_unaligned_le16(swap_endianness16(v), (__be16 *)p->out);
		break;
	case 4:
		put_unaligned_le32(swap_endianness32(v), (__be32 *)p->out);
		break;
	case 8:
		put_unaligned_le64(swap_endianness64(v), (__be64 *)p->out);
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
	int ret;

	ret = next_bits(p, &index, bits);
	if (ret)
		return ret;

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

	if (size != 2 && size != 4 && size != 8)
		printf("__do_index invalid size %x\n", size);
	#ifdef DEBUG
	else	
		printf("index%x to %lx off %lx adjoff %lx tot %lx data %lx\n",
			 size, (unsigned long)index,
			 (unsigned long)(index * size), (unsigned long)offset,
			 (unsigned long)total,
			 (unsigned long)beN_to_cpu(&p->ostart[offset], size));
	#endif

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
	int i, ret = 0;

	if (o >= OPS_MAX)
		return -EINVAL;

	for (i = 0; i < 4; i++) {
		uint8_t op = decomp_ops[o][i];

		#ifdef DEBUG
		printf("op is %x\n", op);
		#endif

		switch (op & OP_ACTION) {
		case OP_ACTION_DATA:
			ret = do_data(p, op & OP_AMOUNT);
			break;
		case OP_ACTION_INDEX:
			ret = do_index(p, op & OP_AMOUNT);
			break;
		case OP_ACTION_NOOP:
			break;
		default:
			fprintf(stderr, "Internal error, invalid op %x\n", op);
			return -EINVAL;
		}

		if (ret)
			return ret;
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
	struct sw842_param_decomp p;
	int ret;
	uint64_t op, rep, tmp, bytes, total;
	uint64_t crc;

	p.in = (uint8_t *)in;
	p.bit = 0;
	p.ilen = ilen;
	p.out = out;
	p.ostart = out;
	p.olen = *olen;

	total = p.olen;

	*olen = 0;

	do {
		ret = next_bits(&p, &op, OP_BITS);
		if (ret)
			return ret;

		#ifdef DEBUG
		printf("template is %lx\n", (unsigned long)op);
		#endif

		switch (op) {
		case OP_REPEAT:
			ret = next_bits(&p, &rep, REPEAT_BITS);
			if (ret)
				return ret;

			if (p.out == out) /* no previous bytes */
				return -EINVAL;

			/* copy rep + 1 */
			rep++;

			if (rep * 8 > p.olen)
				return -ENOSPC;

			while (rep-- > 0) {
				memcpy(p.out, p.out - 8, 8);
				p.out += 8;
				p.olen -= 8;
			}

			break;
		case OP_ZEROS:
			if (8 > p.olen)
				return -ENOSPC;

			memset(p.out, 0, 8);
			p.out += 8;
			p.olen -= 8;

			break;
		case OP_SHORT_DATA:
			ret = next_bits(&p, &bytes, SHORT_DATA_BITS);
			if (ret)
				return ret;

			if (!bytes || bytes > SHORT_DATA_BITS_MAX)
				return -EINVAL;

			while (bytes-- > 0) {
				ret = next_bits(&p, &tmp, 8);
				if (ret)
					return ret;
				*p.out = (uint8_t)tmp;
				p.out++;
				p.olen--;
			}

			break;
		case OP_END:

			break;
		default: /* use template */
			ret = do_op(&p, op);
			if (ret)
				return ret;
			break;
		}
	} while (op != OP_END);

	/*
	 * crc(0:31) is saved in compressed data starting with the
	 * next bit after End of stream template.
	 */
	ret = next_bits(&p, &crc, CRC_BITS);
	if (ret)
		return ret;

	/*
	 * Validate CRC saved in compressed data.
	 *
	if (crc != (uint64_t)crc32_be(0, out, total - p.olen)) {
		fprintf(stderr, "CRC mismatch for decompression\n");
		return -EINVAL;
	}*/

	if (unlikely((total - p.olen) > UINT_MAX))
		return -ENOSPC;

	*olen = total - p.olen;

	return 0;
}


