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
#include <lib842/sw.h>
#include "../common/memaccess.h"
#include "../common/endianness.h"
#include "../common/crc32.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <cerrno>

struct sw842_param_decomp {
	uint8_t *out;
	const uint8_t *ostart;
	const uint64_t *in;
#ifdef ENABLE_ERROR_HANDLING
	const uint64_t *istart;
	size_t ilen;
	size_t olen;
	int errorcode;
#endif
	uint8_t bits;
	uint64_t buffer;
};

/* rolling fifo sizes */
#define I2_FIFO_SIZE (2 * (1 << I2_BITS))
#define I4_FIFO_SIZE (4 * (1 << I4_BITS))
#define I8_FIFO_SIZE (8 * (1 << I8_BITS))

#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define round_down(x, y) ((x) & ~__round_mask(x, y))

/* number of bits in a buffered word */
#define WSIZE 64 //sizeof(uint64_t)

#if defined(BRANCH_FREE) && BRANCH_FREE == 1
static const uint16_t fifo_sizes[9] = { 0, 0, I2_FIFO_SIZE, 0, I4_FIFO_SIZE, 0,
				        0, 0, I8_FIFO_SIZE };

static const uint8_t dec_templates[26][4][2] = {
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
#endif

/* read a single uint64_t from memory */
static inline uint64_t read_word(struct sw842_param_decomp *p)
{
#ifdef ENABLE_ERROR_HANDLING
	if ((p->in - p->istart + 1) * sizeof(uint64_t) > p->ilen) {
		if (p->errorcode == 0)
			p->errorcode = -EINVAL;
		return 0;
	}
#endif
	uint64_t w = swap_be_to_native64(*p->in++);
	return w;
}

/* read 0 <= n <= 64 bits */
static inline uint64_t read_bits(struct sw842_param_decomp *p, uint8_t n)
{
	uint64_t value = p->buffer >> (WSIZE - n);
	value &= (n > 0) ? 0xFFFFFFFFFFFFFFFF : 0x0000000000000000;

	if (p->bits < n) {
		/* fetch WSIZE bits  */
		p->buffer = read_word(p);
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

#if (defined(BRANCH_FREE) && BRANCH_FREE == 0) || not defined(BRANCH_FREE)
template <uint8_t N> static inline void do_data(struct sw842_param_decomp *p)
{
#ifdef ENABLE_ERROR_HANDLING
	if (static_cast<size_t>(p->out - p->ostart + N) > p->olen) {
		if (p->errorcode == 0)
			p->errorcode = -ENOSPC;
		return;
	}
#endif

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

static inline void do_index(struct sw842_param_decomp *p, uint8_t size,
			    uint8_t bits, uint64_t fsize)
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

#ifdef ENABLE_ERROR_HANDLING
	if (offset + size > total) {
		if (p->errorcode == 0) {
			fprintf(stderr, "index%x %lx points past end %lx\n", size,
				(unsigned long)offset, (unsigned long)total);
			p->errorcode = -EINVAL;
		}
		return;
	}
#endif

#ifdef ENABLE_ERROR_HANDLING
	if (static_cast<size_t>(p->out - p->ostart + size) > p->olen) {
		if (p->errorcode == 0)
			p->errorcode = -ENOSPC;
		return;
	}
#endif

	memcpy(p->out, &p->ostart[offset], size);
	p->out += size;
}
#endif

#if defined(BRANCH_FREE) && BRANCH_FREE == 1
static inline uint64_t get_index(struct sw842_param_decomp *p,
				 uint8_t size, uint64_t index, uint64_t fsize)
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

#ifdef ENABLE_ERROR_HANDLING
	if (offset + size > total) {
		if (p->errorcode == 0) {
			fprintf(stderr, "index%x %lx points past end %lx\n", size,
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
	uint64_t output_word = 0;
	uint64_t values[8] = { 0 };
	uint8_t bits = 0;

#ifdef ENABLE_ERROR_HANDLING
	if (op >= OPS_MAX) {
		p->errorcode = -EINVAL;
		return;
	}
#endif
	for (int i = 0; i < 4; i++) {
		// 0-initialize all values-fields
		values[i] = 0;
		values[4 + i] = 0;

		uint8_t dec_template = dec_templates[op][i][0];
		uint8_t is_index = (dec_template >> 7);
		uint8_t num_bits = dec_template & 0x7F;
		uint8_t dst_size = dec_templates[op][i][1];

		values[(4 * is_index) + i] =
			read_bits(p, num_bits);
#ifdef ENABLE_ERROR_HANDLING
		if (p->errorcode != 0)
			return;
#endif

#ifdef ENABLE_ERROR_HANDLING
		uint64_t offset = is_index ? get_index(p, dst_size,
			values[4 + i], fifo_sizes[dst_size]) : 0;
		if (p->errorcode != 0)
			return;
#else
		uint64_t offset = get_index(p, dst_size,
			values[4 + i], fifo_sizes[dst_size]);
#endif
		memcpy(&values[4 + i],
		       &p->ostart[offset * is_index], dst_size * is_index);
		values[4 + i] = swap_be_to_native64(
			values[4 + i] << (WSIZE - (dst_size << 3)));

		values[i] = values[4 + i] * is_index | values[i];
		output_word |= values[i] << (64 - (dst_size << 3) - bits);
		bits += dst_size << 3;
	}
#ifdef ENABLE_ERROR_HANDLING
	if (p->out - p->ostart + 8 > p->olen) {
		p->errorcode = -ENOSPC;
		return;
	}
#endif
	write64(p->out, swap_native_to_be64(output_word));
	p->out += 8;
}
#endif

/**
 * optsw842_decompress
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
int optsw842_decompress(const uint8_t *in, size_t ilen,
			uint8_t *out, size_t *olen)
{
	struct sw842_param_decomp p;
	p.out = out;
	p.ostart = out;
	p.in = (const uint64_t *)in;
#ifdef ENABLE_ERROR_HANDLING
	p.istart = p.in;
	p.ilen = ilen;
	p.olen = *olen;
	p.errorcode = 0;
#endif
	p.buffer = 0;
	p.bits = 0;

	*olen = 0;

	uint64_t op, rep;

	do {
		op = read_bits(&p, OP_BITS);
#ifdef ENABLE_ERROR_HANDLING
		if (p.errorcode != 0)
			return p.errorcode;
#endif

#ifdef DEBUG
		printf("template is %llx\n", op);
#endif

		switch (op) {
#if (defined(BRANCH_FREE) && BRANCH_FREE == 0) || not defined(BRANCH_FREE)
		case 0x00: // { D8, N0, N0, N0 }, 64 bits
			do_data<8>(&p);
			break;
		case 0x01: // { D4, D2, I2, N0 }, 56 bits
			do_data<4>(&p);
			do_data<2>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			break;
		case 0x02: // { D4, I2, D2, N0 }, 56 bits
			do_data<4>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<2>(&p);
			break;
		case 0x03: // { D4, I2, I2, N0 }, 48 bits
			do_data<4>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			break;
		case 0x04: // { D4, I4, N0, N0 }, 41 bits
			do_data<4>(&p);
			do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
			break;
		case 0x05: // { D2, I2, D4, N0 }, 56 bits
			do_data<2>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<4>(&p);
			break;
		case 0x06: // { D2, I2, D2, I2 }, 48 bits
			do_data<2>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<2>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			break;
		case 0x07: // { D2, I2, I2, D2 }, 48 bits
			do_data<2>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<2>(&p);
			break;
		case 0x08: // { D2, I2, I2, I2 }, 40 bits
			do_data<2>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			break;
		case 0x09: // { D2, I2, I4, N0 }, 33 bits
			do_data<2>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
			break;
		case 0x0a: // { I2, D2, D4, N0 }, 56 bits
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<2>(&p);
			do_data<4>(&p);
			break;
		case 0x0b: // { I2, D4, I2, N0 }, 48 bits
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<4>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			break;
		case 0x0c: // { I2, D2, I2, D2 }, 48 bits
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<2>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<2>(&p);
			break;
		case 0x0d: // { I2, D2, I2, I2 }, 40 bits
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<2>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			break;
		case 0x0e: // { I2, D2, I4, N0 }, 33 bits
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<2>(&p);
			do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
			break;
		case 0x0f: // { I2, I2, D4, N0 }, 48 bits
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<4>(&p);
			break;
		case 0x10: // { I2, I2, D2, I2 }, 40 bits
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<2>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			break;
		case 0x11: // { I2, I2, I2, D2 }, 40 bits
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<2>(&p);
			break;
		case 0x12: // { I2, I2, I2, I2 }, 32 bits
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			break;
		case 0x13: // { I2, I2, I4, N0 }, 25 bits
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
			break;
		case 0x14: // { I4, D4, N0, N0 }, 41 bits
			do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
			do_data<4>(&p);
			break;
		case 0x15: // { I4, D2, I2, N0 }, 33 bits
			do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
			do_data<2>(&p);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			break;
		case 0x16: // { I4, I2, D2, N0 }, 33 bits
			do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_data<2>(&p);
			break;
		case 0x17: // { I4, I2, I2, N0 }, 25 bits
			do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			do_index(&p, 2, I2_BITS, I2_FIFO_SIZE);
			break;
		case 0x18: // { I4, I4, N0, N0 }, 18 bits
			do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
			do_index(&p, 4, I4_BITS, I4_FIFO_SIZE);
			break;
		case 0x19: // { I8, N0, N0, N0 }, 8 bits
			do_index(&p, 8, I8_BITS, I8_FIFO_SIZE);
			break;
#endif
		case OP_REPEAT:
			rep = read_bits(&p, REPEAT_BITS);
#ifdef ENABLE_ERROR_HANDLING
			if (p.errorcode != 0)
				return p.errorcode;

			if (p.out == out) /* no previous bytes */
				return -EINVAL;
#endif

			/* copy rep + 1 */
			rep++;

#ifdef ENABLE_ERROR_HANDLING
			if (p.out - p.ostart + rep * 8 > p.olen)
				return -ENOSPC;
#endif

			while (rep-- > 0) {
				memcpy(p.out, p.out - 8, 8);
				p.out += 8;
			}
			break;
		case OP_ZEROS:
#ifdef ENABLE_ERROR_HANDLING
			if (static_cast<size_t>(p.out - p.ostart + 8) > p.olen)
				return -ENOSPC;
#endif

			memset(p.out, 0, 8);
			p.out += 8;
			break;
		case OP_END:
			break;
		default:
#if defined(BRANCH_FREE) && BRANCH_FREE == 1
			do_op(&p, op);
#ifdef ENABLE_ERROR_HANDLING
			if (p.errorcode != 0)
				return p.errorcode;
#endif
#else
			fprintf(stderr, "Invalid op template: %" PRIx64 "\n", op);
			return -EINVAL;
#endif
		}


#ifdef ENABLE_ERROR_HANDLING
		if (p.errorcode != 0)
			return p.errorcode;
#endif
	} while (op != OP_END);

	/*
	 * crc(0:31) is saved in compressed data starting with the
	 * next bit after End of stream template.
	 */
#ifndef DISABLE_CRC
	uint64_t crc = read_bits(&p, CRC_BITS);
#ifdef ENABLE_ERROR_HANDLING
	if (p.errorcode != 0)
		return p.errorcode;
#endif

	/*
	 * Validate CRC saved in compressed data.
	 */
	if (crc != (uint64_t)crc32_be(0, out, p.out - p.ostart)) {
		fprintf(stderr, "CRC mismatch for decompression\n");
		return -EINVAL;
	}
#endif

	*olen = p.out - p.ostart;

	return 0;
}
