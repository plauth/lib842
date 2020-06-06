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

/* read 0 < n <= 64 bits */
static inline uint64_t read_bits(struct sw842_param_decomp *p, uint8_t n)
{
	uint64_t value = p->buffer >> (WSIZE - n);
	if (p->bits < n) {
		/* fetch WSIZE bits  */
		p->buffer = read_word(p);
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

#if (defined(BRANCH_FREE) && BRANCH_FREE == 0) || not defined(BRANCH_FREE)
template <uint8_t N> static inline void do_data(struct sw842_param_decomp *p,
						uint64_t data)
{
	switch (N) {
	case 2:
		write16(p->out, swap_be_to_native16(data));
		break;
	case 4:
		write32(p->out, swap_be_to_native32(data));
		break;
	case 8:
		write64(p->out, swap_be_to_native64(data));
		break;
	}

	p->out += N;
}

static inline void do_index(struct sw842_param_decomp *p, uint8_t size,
			    uint64_t index, uint64_t fsize)
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
		fprintf(stderr, "index%x %lx points past end %lx\n", size,
			(unsigned long)offset, (unsigned long)total);
		p->errorcode = -EINVAL;
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

	// TODOXXX explain the patterns those formulas are based on
	uint8_t opbits = 64 - ((op % 5) + 1) / 2 * 8 - ((op % 5) / 4) * 7
			    - ((op / 5) + 1) / 2 * 8 - ((op / 5) / 4) * 7;
	uint64_t params = read_bits(p, opbits);
#ifdef ENABLE_ERROR_HANDLING
	if (p->errorcode != 0)
		return;
#endif

	for (int i = 0; i < 4; i++) {
		// 0-initialize all values-fields
		values[i] = 0;
		values[4 + i] = 0;

		// TODOXXX explain the patterns those formulas are based on
		uint8_t opchunk = (i < 2) ? op / 5 : op % 5;
		uint32_t is_index = (i & 1) * (opchunk & 1) + ((i & 1) ^ 1) * (opchunk >= 2);
		uint32_t dst_size = 2 + (opchunk >= 4) * (1 - 2 * (i % 2)) * 2;
		uint8_t num_bits = (i & 1) * (16 - (opchunk % 2) * 8 - (opchunk >= 4) * 16) +
				   ((i & 1) ^ 1) * (16 - (opchunk / 2) * 8 + (opchunk >= 4) * 9);

		// https://stackoverflow.com/a/28703383
		uint64_t bitsmask = ((uint64_t)-(num_bits != 0)) &
				    (((uint64_t)-1) >> (64 - num_bits));
		values[(4 * is_index) + i] =
			(params >> (opbits - num_bits)) & bitsmask;
		opbits -= num_bits;

		// TODOXXX explain how this relates to In_FIFO_SIZE constants
#ifdef ENABLE_ERROR_HANDLING
		uint64_t offset = is_index ? get_index(p, dst_size,
			values[4 + i], 2048 - 1536 * ((dst_size >> 2) < 1)) : 0;
		if (p->errorcode != 0)
			return;
#else
		uint64_t offset = get_index(p, dst_size,
			values[4 + i], 2048 - 1536 * ((dst_size >> 2) < 1));
#endif
		memcpy(&values[4 + i],
		       &p->ostart[offset * is_index], dst_size * is_index);
		values[4 + i] = swap_be_to_native64(
			values[4 + i]) >> (WSIZE - (dst_size << 3));

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

#ifdef ENABLE_ERROR_HANDLING
		// All valid ops except OP_REPEAT and OP_END generate exactly
		// 8 bytes of output, so check the buffer capacity here for
		// all of those instead of doing it separately for each opcode
		if ((op < OPS_MAX || op == OP_ZEROS) &&
		    static_cast<size_t>(p.out - p.ostart + 8) > p.olen)
			return -ENOSPC;
#endif

		switch (op) {
#if (defined(BRANCH_FREE) && BRANCH_FREE == 0) || not defined(BRANCH_FREE)
		case 0x00: // { D8, N0, N0, N0 }, 64 bits
			rep = read_bits(&p, 64);
			do_data<8>(&p, rep);
			break;
		case 0x01: // { D4, D2, I2, N0 }, 56 bits
			rep = read_bits(&p, 32 + 16 + I2_BITS);
			do_data<4>(&p, (rep >> 24) & ((1ULL << 32) - 1));
			do_data<2>(&p, (rep >> 8) & ((1ULL << 16) - 1));
			do_index(&p, 2, (rep >> 0) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			break;
		case 0x02: // { D4, I2, D2, N0 }, 56 bits
			rep = read_bits(&p, 32 + I2_BITS + 16);
			do_data<4>(&p, (rep >> 24) & ((1ULL << 32) - 1));
			do_index(&p, 2, (rep >> 16) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<2>(&p, (rep >> 0) & ((1ULL << 16) - 1));
			break;
		case 0x03: // { D4, I2, I2, N0 }, 48 bits
			rep = read_bits(&p, 32 + I2_BITS + I2_BITS);
			do_data<4>(&p, (rep >> 16) & ((1ULL << 32) - 1));
			do_index(&p, 2, (rep >> 8) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 0) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			break;
		case 0x04: // { D4, I4, N0, N0 }, 41 bits
			rep = read_bits(&p, 32 + I4_BITS);
			do_data<4>(&p, (rep >> 9) & ((1ULL << 32) - 1));
			do_index(&p, 4, (rep >> 0) & ((1ULL << I4_BITS) - 1), I4_FIFO_SIZE);
			break;
		case 0x05: // { D2, I2, D4, N0 }, 56 bits
			rep = read_bits(&p, 16 + I2_BITS + 32);
			do_data<2>(&p, (rep >> 40) & ((1ULL << 16) - 1));
			do_index(&p, 2, (rep >> 32) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<4>(&p, (rep >> 0) & ((1ULL << 32) - 1));
			break;
		case 0x06: // { D2, I2, D2, I2 }, 48 bits
			rep = read_bits(&p, 16 + I2_BITS + 16 + I2_BITS);
			do_data<2>(&p, (rep >> 32) & ((1ULL << 16) - 1));
			do_index(&p, 2, (rep >> 24) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<2>(&p, (rep >> 8) & ((1ULL << 16) - 1));
			do_index(&p, 2, (rep >> 0) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			break;
		case 0x07: // { D2, I2, I2, D2 }, 48 bits
			rep = read_bits(&p, 16 + I2_BITS + I2_BITS + 16);
			do_data<2>(&p, (rep >> 32) & ((1ULL << 16) - 1));
			do_index(&p, 2, (rep >> 24) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 16) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<2>(&p, (rep >> 0) & ((1ULL << 16) - 1));
			break;
		case 0x08: // { D2, I2, I2, I2 }, 40 bits
			rep = read_bits(&p, 16 + I2_BITS + I2_BITS + I2_BITS);
			do_data<2>(&p, (rep >> 24) & ((1ULL << 16) - 1));
			do_index(&p, 2, (rep >> 16) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 8) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 0) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			break;
		case 0x09: // { D2, I2, I4, N0 }, 33 bits
			rep = read_bits(&p, 16 + I2_BITS + I4_BITS);
			do_data<2>(&p, (rep >> 17) & ((1ULL << 16) - 1));
			do_index(&p, 2, (rep >> 9) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 4, (rep >> 0) & ((1ULL << I4_BITS) - 1), I4_FIFO_SIZE);
			break;
		case 0x0a: // { I2, D2, D4, N0 }, 56 bits
			rep = read_bits(&p, I2_BITS + 16 + 32);
			do_index(&p, 2, (rep >> 48) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<2>(&p, (rep >> 32) & ((1ULL << 16) - 1));
			do_data<4>(&p, (rep >> 0) & ((1ULL << 32) - 1));
			break;
		case 0x0b: // { I2, D4, I2, N0 }, 48 bits
			rep = read_bits(&p, I2_BITS + 32 + I2_BITS);
			do_index(&p, 2, (rep >> 40) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<4>(&p, (rep >> 8) & ((1ULL << 32) - 1));
			do_index(&p, 2, (rep >> 0) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			break;
		case 0x0c: // { I2, D2, I2, D2 }, 48 bits
			rep = read_bits(&p, I2_BITS + 16 + I2_BITS + 16);
			do_index(&p, 2, (rep >> 40) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<2>(&p, (rep >> 24) & ((1ULL << 16) - 1));
			do_index(&p, 2, (rep >> 16) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<2>(&p, (rep >> 0) & ((1ULL << 16) - 1));
			break;
		case 0x0d: // { I2, D2, I2, I2 }, 40 bits
			rep = read_bits(&p, I2_BITS + 16 + I2_BITS + I2_BITS);
			do_index(&p, 2, (rep >> 32) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<2>(&p, (rep >> 16) & ((1ULL << 16) - 1));
			do_index(&p, 2, (rep >> 8) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 0) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			break;
		case 0x0e: // { I2, D2, I4, N0 }, 33 bits
			rep = read_bits(&p, I2_BITS + 16 + I4_BITS);
			do_index(&p, 2, (rep >> 25) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<2>(&p, (rep >> 9) & ((1ULL << 16) - 1));
			do_index(&p, 4, (rep >> 0) & ((1ULL << I4_BITS) - 1), I4_FIFO_SIZE);
			break;
		case 0x0f: // { I2, I2, D4, N0 }, 48 bits
			rep = read_bits(&p, I2_BITS + I2_BITS + 32);
			do_index(&p, 2, (rep >> 40) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 32) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<4>(&p, (rep >> 0) & ((1ULL << 32) - 1));
			break;
		case 0x10: // { I2, I2, D2, I2 }, 40 bits
			rep = read_bits(&p, I2_BITS + I2_BITS + 16 + I2_BITS);
			do_index(&p, 2, (rep >> 32) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 24) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<2>(&p, (rep >> 8) & ((1ULL << 16) - 1));
			do_index(&p, 2, (rep >> 0) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			break;
		case 0x11: // { I2, I2, I2, D2 }, 40 bits
			rep = read_bits(&p, I2_BITS + I2_BITS + I2_BITS + 16);
			do_index(&p, 2, (rep >> 32) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 24) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 16) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<2>(&p, (rep >> 0) & ((1ULL << 16) - 1));
			break;
		case 0x12: // { I2, I2, I2, I2 }, 32 bits
			rep = read_bits(&p, I2_BITS + I2_BITS + I2_BITS + I2_BITS);
			do_index(&p, 2, (rep >> 24) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 16) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 8) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 0) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			break;
		case 0x13: // { I2, I2, I4, N0 }, 25 bits
			rep = read_bits(&p, I2_BITS + I2_BITS + I4_BITS);
			do_index(&p, 2, (rep >> 17) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 9) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 4, (rep >> 0) & ((1ULL << I4_BITS) - 1), I4_FIFO_SIZE);
			break;
		case 0x14: // { I4, D4, N0, N0 }, 41 bits
			rep = read_bits(&p, I4_BITS + 32);
			do_index(&p, 4, (rep >> 32) & ((1ULL << I4_BITS) - 1), I4_FIFO_SIZE);
			do_data<4>(&p, (rep >> 0) & ((1ULL << 32) - 1));
			break;
		case 0x15: // { I4, D2, I2, N0 }, 33 bits
			rep = read_bits(&p, I4_BITS + 16 + I2_BITS);
			do_index(&p, 4, (rep >> 24) & ((1ULL << I4_BITS) - 1), I4_FIFO_SIZE);
			do_data<2>(&p, (rep >> 8) & ((1ULL << 16) - 1));
			do_index(&p, 2, (rep >> 0) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			break;
		case 0x16: // { I4, I2, D2, N0 }, 33 bits
			rep = read_bits(&p, I4_BITS + I2_BITS + 16);
			do_index(&p, 4, (rep >> 24) & ((1ULL << I4_BITS) - 1), I4_FIFO_SIZE);
			do_index(&p, 2, (rep >> 16) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_data<2>(&p, (rep >> 0) & ((1ULL << 16) - 1));
			break;
		case 0x17: // { I4, I2, I2, N0 }, 25 bits
			rep = read_bits(&p, I4_BITS + I2_BITS + I2_BITS);
			do_index(&p, 4, (rep >> 16) & ((1ULL << I4_BITS) - 1), I4_FIFO_SIZE);
			do_index(&p, 2, (rep >> 8) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			do_index(&p, 2, (rep >> 0) & ((1ULL << I2_BITS) - 1), I2_FIFO_SIZE);
			break;
		case 0x18: // { I4, I4, N0, N0 }, 18 bits
			rep = read_bits(&p, I4_BITS + I4_BITS);
			do_index(&p, 4, (rep >> 9) & ((1ULL << I4_BITS) - 1), I4_FIFO_SIZE);
			do_index(&p, 4, (rep >> 0) & ((1ULL << I4_BITS) - 1), I4_FIFO_SIZE);
			break;
		case 0x19: // { I8, N0, N0, N0 }, 8 bits
			rep = read_bits(&p, I8_BITS);
			do_index(&p, 8, (rep >> 0) & ((1ULL << I8_BITS) - 1), I8_FIFO_SIZE);
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
			memset(p.out, 0, 8);
			p.out += 8;
			break;
		case OP_END:
			break;
#if defined(BRANCH_FREE) && BRANCH_FREE == 1
		case (OPS_MAX - 1): {
			uint64_t value = read_bits(&p, 8);
#ifdef ENABLE_ERROR_HANDLING
			if (p.errorcode != 0)
				return p.errorcode;
#endif
			uint64_t offset = get_index(&p, 8, value, I8_FIFO_SIZE);
#ifdef ENABLE_ERROR_HANDLING
			if (p.errorcode != 0)
				return p.errorcode;
#endif
			memcpy(&value, &p.ostart[offset], 8);
			write64(p.out, value);
			p.out += 8;
		}
		break;
		default:
			do_op(&p, op);
#ifdef ENABLE_ERROR_HANDLING
			if (p.errorcode != 0)
				return p.errorcode;
#endif
#else
		default:
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
