/*
 * 842 Software Compression
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

/* By default, we allow compressing input buffers of any length, but we must
 * use the non-standard "short data" template so the decompressor can correctly
 * reproduce the uncompressed data buffer at the right length.  However the
 * hardware 842 compressor will not recognize the "short data" template, and
 * will fail to decompress any compressed buffer containing it (I have no idea
 * why anyone would want to use software to compress and hardware to decompress
 * but that's beside the point).  This parameter forces the compression
 * function to simply reject any input buffer that isn't a multiple of 8 bytes
 * long, instead of using the "short data" template, so that all compressed
 * buffers produced by this function will be decompressable by the 842 hardware
 * decompressor.  Unless you have a specific need for that, leave this disabled
 * so that any length buffer can be compressed.
 */
#define SW842_STRICT 1

static uint8_t comp_ops[OPS_MAX][5] = { /* params size in bits */
	{ I8, N0, N0, N0, 0x19 }, /* 8 */
	{ I4, I4, N0, N0, 0x18 }, /* 18 */
	{ I4, I2, I2, N0, 0x17 }, /* 25 */
	{ I2, I2, I4, N0, 0x13 }, /* 25 */
	{ I2, I2, I2, I2, 0x12 }, /* 32 */
	{ I4, I2, D2, N0, 0x16 }, /* 33 */
	{ I4, D2, I2, N0, 0x15 }, /* 33 */
	{ I2, D2, I4, N0, 0x0e }, /* 33 */
	{ D2, I2, I4, N0, 0x09 }, /* 33 */
	{ I2, I2, I2, D2, 0x11 }, /* 40 */
	{ I2, I2, D2, I2, 0x10 }, /* 40 */
	{ I2, D2, I2, I2, 0x0d }, /* 40 */
	{ D2, I2, I2, I2, 0x08 }, /* 40 */
	{ I4, D4, N0, N0, 0x14 }, /* 41 */
	{ D4, I4, N0, N0, 0x04 }, /* 41 */
	{ I2, I2, D4, N0, 0x0f }, /* 48 */
	{ I2, D2, I2, D2, 0x0c }, /* 48 */
	{ I2, D4, I2, N0, 0x0b }, /* 48 */
	{ D2, I2, I2, D2, 0x07 }, /* 48 */
	{ D2, I2, D2, I2, 0x06 }, /* 48 */
	{ D4, I2, I2, N0, 0x03 }, /* 48 */
	{ I2, D2, D4, N0, 0x0a }, /* 56 */
	{ D2, I2, D4, N0, 0x05 }, /* 56 */
	{ D4, I2, D2, N0, 0x02 }, /* 56 */
	{ D4, D2, I2, N0, 0x01 }, /* 56 */
	{ D8, N0, N0, N0, 0x00 }, /* 64 */
};

//static uint64_t outbits = 0;

#define INDEX_NOT_FOUND		(-1)
#define INDEX_NOT_CHECKED	(-2)

#define get_input_data(p, o, b)						\
	swap_endianness##b(get_unaligned##b((__be##b *)((p)->in + (o))))

#define UINT_TYPE(b) UINT_TYPE_##b
#define UINT_TYPE_2 uint16_t
#define UINT_TYPE_4 uint32_t
#define UINT_TYPE_8 uint64_t

#define find_index(p, b, n)	({					\
	p->index##b[n] = INDEX_NOT_FOUND;			\
	UINT_TYPE(b) _n = p->data##b[n];			\
												\
	struct hlist_node##b *h;					\
	HASH_FIND(hh, p->htable##b, &_n, b, h);		\
	if(h != NULL && h->head != NULL)			\
		p->index##b[n] = h->head->index;		\
	p->index##b[n] >= 0;						\
})

#define check_index(p, b, n) ((p)->index##b[n] == INDEX_NOT_CHECKED ? find_index(p, b, n) : (p)->index##b[n] >= 0)

#define replace_hash(p, b, i, d)	do {													\
	int node_index = i+d;																	\
	UINT_TYPE(b) _n = p->node##b[node_index];												\
																							\
	struct hlist_node##b *h;																\
	HASH_FIND(hh, p->htable##b, &_n, b, h);													\
																							\
	if(h != NULL) {																			\
		struct node##b##_el *el;														\
		DL_SEARCH_SCALAR(h->head,el,index,node_index);										\
		if(el != NULL) {																	\
			DL_DELETE(h->head,el);															\
      		free(el);																		\
		}																					\
	}																						\
																							\
	h = NULL;																				\
	_n = p->data##b[d];																		\
	p->node##b[node_index] = _n;															\
																							\
	HASH_FIND(hh, p->htable##b, &_n, b, h);													\
	if(h == NULL) {																			\
		h = (struct hlist_node##b *) malloc(sizeof(struct hlist_node##b));					\
    	h->data = _n;																		\
    	h->head = NULL;																		\
    	HASH_ADD(hh, p->htable##b, data, b, h);												\
    }																						\
    																						\
    struct node##b##_el *el = (struct node##b##_el *) malloc(sizeof(struct node##b##_el));	\
    el->index = node_index;																	\
    DL_APPEND(h->head, el);																	\
} while (0)

static uint8_t bmask[8] = { 0x00, 0x80, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc, 0xfe };

static int add_bits(struct sw842_param *p, uint64_t d, uint8_t n);

static int __split_add_bits(struct sw842_param *p, uint64_t d, uint8_t n, uint8_t s)
{
	int ret;

	if (n <= s)
		return -EINVAL;

	ret = add_bits(p, d >> s, n - s);
	if (ret)
		return ret;
	return add_bits(p, d & GENMASK_ULL(s - 1, 0), s);
}

static int add_bits(struct sw842_param *p, uint64_t d, uint8_t n)
{
	int b = p->bit, bits = b + n, s = round_up(bits, 8) - bits;
	uint64_t o;
	uint8_t *out = p->out;

	#ifdef DEBUG
	printf("add %u bits %lx\n", (unsigned char)n, (unsigned long)d);
	#endif

	if (n > 64)
		return -EINVAL;

	/* split this up if writing to > 8 bytes (i.e. n == 64 && p->bit > 0),
	 * or if we're at the end of the output buffer and would write past end
	 */
	if (bits > 64)
		return __split_add_bits(p, d, n, 32);
	else if (p->olen < 8 && bits > 32 && bits <= 56)
		return __split_add_bits(p, d, n, 16);
	else if (p->olen < 4 && bits > 16 && bits <= 24)
		return __split_add_bits(p, d, n, 8);

	if (DIV_ROUND_UP(bits, 8) > p->olen)
		return -ENOSPC;
 
	//outbits += n;
	o = *out & bmask[b];
	d <<= s;

	if (bits <= 8)
		*out = o | d;
	else if (bits <= 16)
		put_unaligned_le16(swap_endianness16(o << 8 | d), (__be16 *)out);
	else if (bits <= 24)
		put_unaligned_le32(swap_endianness32(o << 24 | d << 8), (__be32 *)out);
	else if (bits <= 32)
		put_unaligned_le32(swap_endianness32(o << 24 | d), (__be32 *)out);
	else if (bits <= 40)
		put_unaligned_le64(swap_endianness64(o << 56 | d << 24), (__be64 *)out);
	else if (bits <= 48)
		put_unaligned_le64(swap_endianness64(o << 56 | d << 16), (__be64 *)out);
	else if (bits <= 56)
		put_unaligned_le64(swap_endianness64(o << 56 | d << 8), (__be64 *)out);
	else
		put_unaligned_le64(swap_endianness64(o << 56 | d), (__be64 *)out);

	p->bit += n;

	if (p->bit > 7) {
		p->out += p->bit / 8;
		p->olen -= p->bit / 8;
		p->bit %= 8;
	}

	return 0;
}

static int add_template(struct sw842_param *p, uint8_t c)
{
	int ret, i, b = 0;
	uint8_t *t = comp_ops[c];
	bool inv = false;

	if (c >= OPS_MAX)
		return -EINVAL;

	#ifdef DEBUG
	printf("template %x\n", t[4]);
	#endif

	ret = add_bits(p, t[4], OP_BITS);
	if (ret)
		return ret;

	for (i = 0; i < 4; i++) {
		#ifdef DEBUG
		printf("op %x\n", t[i]);
		#endif

		switch (t[i] & OP_AMOUNT) {
		case OP_AMOUNT_8:
			if (b)
				inv = true;
			else if (t[i] & OP_ACTION_INDEX)
				ret = add_bits(p, p->index8[0], I8_BITS);
			else if (t[i] & OP_ACTION_DATA)
				ret = add_bits(p, p->data8[0], 64);
			else
				inv = true;
			break;
		case OP_AMOUNT_4:
			if (b == 2 && t[i] & OP_ACTION_DATA)
				ret = add_bits(p, get_input_data(p, 2, 32), 32);
			else if (b != 0 && b != 4)
				inv = true;
			else if (t[i] & OP_ACTION_INDEX)
				ret = add_bits(p, p->index4[b >> 2], I4_BITS);
			else if (t[i] & OP_ACTION_DATA)
				ret = add_bits(p, p->data4[b >> 2], 32);
			else
				inv = true;
			break;
		case OP_AMOUNT_2:
			if (b != 0 && b != 2 && b != 4 && b != 6)
				inv = true;
			if (t[i] & OP_ACTION_INDEX)
				ret = add_bits(p, p->index2[b >> 1], I2_BITS);
			else if (t[i] & OP_ACTION_DATA)
				ret = add_bits(p, p->data2[b >> 1], 16);
			else
				inv = true;
			break;
		case OP_AMOUNT_0:
			inv = (b != 8) || !(t[i] & OP_ACTION_NOOP);
			break;
		default:
			inv = true;
			break;
		}

		if (ret)
			return ret;

		if (inv) {
			fprintf(stderr, "Invalid templ %x op %d : %x %x %x %x\n",
			       c, i, t[0], t[1], t[2], t[3]);
			return -EINVAL;
		}

		b += t[i] & OP_AMOUNT;
	}

	if (b != 8) {
		fprintf(stderr, "Invalid template %x len %x : %x %x %x %x\n",
		       c, b, t[0], t[1], t[2], t[3]);
		return -EINVAL;
	}


	return 0;
}

static int add_repeat_template(struct sw842_param *p, uint8_t r)
{
	int ret;

	/* repeat param is 0-based */
	if (!r || --r > REPEAT_BITS_MAX)
		return -EINVAL;

	ret = add_bits(p, OP_REPEAT, OP_BITS);
	if (ret)
		return ret;

	ret = add_bits(p, r, REPEAT_BITS);
	if (ret)
		return ret;

	return 0;
}

static int add_short_data_template(struct sw842_param *p, uint8_t b)
{
	int ret, i;

	if (!b || b > SHORT_DATA_BITS_MAX)
		return -EINVAL;

	ret = add_bits(p, OP_SHORT_DATA, OP_BITS);
	if (ret)
		return ret;

	ret = add_bits(p, b, SHORT_DATA_BITS);
	if (ret)
		return ret;

	for (i = 0; i < b; i++) {
		ret = add_bits(p, p->in[i], 8);
		if (ret)
			return ret;
	}

	return 0;
}

static int add_zeros_template(struct sw842_param *p)
{
	int ret = add_bits(p, OP_ZEROS, OP_BITS);

	if (ret)
		return ret;

	return 0;
}

static int add_end_template(struct sw842_param *p)
{
	int ret = add_bits(p, OP_END, OP_BITS);

	if (ret)
		return ret;

	return 0;
}

static bool check_template(struct sw842_param *p, uint8_t c)
{
	uint8_t *t = comp_ops[c];
	int i, match, b = 0;

	if (c >= OPS_MAX)
		return false;

	for (i = 0; i < 4; i++) {
		if (t[i] & OP_ACTION_INDEX) {
			if (t[i] & OP_AMOUNT_2)
				match = check_index(p, 2, b >> 1);
			else if (t[i] & OP_AMOUNT_4)
				match = check_index(p, 4, b >> 2);
			else if (t[i] & OP_AMOUNT_8)
				match = check_index(p, 8, 0);
			else
				return false;
			if (!match)
				return false;
		}

		b += t[i] & OP_AMOUNT;
	}

	return true;
}

static void get_next_data(struct sw842_param *p)
{
	p->data8[0] = get_input_data(p, 0, 64);
	p->data4[0] = get_input_data(p, 0, 32);
	p->data4[1] = get_input_data(p, 4, 32);
	p->data2[0] = get_input_data(p, 0, 16);
	p->data2[1] = get_input_data(p, 2, 16);
	p->data2[2] = get_input_data(p, 4, 16);
	p->data2[3] = get_input_data(p, 6, 16);
}

/* update the hashtable entries.
 * only call this after finding/adding the current template
 * the dataN fields for the current 8 byte block must be already updated
 */
static void update_hashtables(struct sw842_param *p)
{
	uint64_t pos = p->in - p->instart;
	uint64_t n8 = (pos >> 3) % (1 << I8_BITS);
	uint64_t n4 = (pos >> 2) % (1 << I4_BITS);
	uint64_t n2 = (pos >> 1) % (1 << I2_BITS);

	replace_hash(p, 8, n8, 0);
	replace_hash(p, 4, n4, 0);
	replace_hash(p, 4, n4, 1);
	replace_hash(p, 2, n2, 0);
	replace_hash(p, 2, n2, 1);
	replace_hash(p, 2, n2, 2);
	replace_hash(p, 2, n2, 3);
}

/* find the next template to use, and add it
 * the p->dataN fields must already be set for the current 8 byte block
 */
static int process_next(struct sw842_param *p)
{
	int ret, i;

	p->index8[0] = INDEX_NOT_CHECKED;
	p->index4[0] = INDEX_NOT_CHECKED;
	p->index4[1] = INDEX_NOT_CHECKED;
	p->index2[0] = INDEX_NOT_CHECKED;
	p->index2[1] = INDEX_NOT_CHECKED;
	p->index2[2] = INDEX_NOT_CHECKED;
	p->index2[3] = INDEX_NOT_CHECKED;

	/* check up to OPS_MAX - 1; last op is our fallback */
	for (i = 0; i < OPS_MAX - 1; i++) {
		if (check_template(p, i))
			break;
	}
	ret = add_template(p, i);
	if (ret)
		return ret;

	return 0;
}

/**
 * sw842_compress
 *
 * Compress the uncompressed buffer of length @ilen at @in to the output buffer
 * @out, using no more than @olen bytes, using the 842 compression format.
 *
 * Returns: 0 on success, error on failure.  The @olen parameter
 * will contain the number of output bytes written on success, or
 * 0 on error.
 */
int sw842_compress(const uint8_t *in, unsigned int ilen,
		   uint8_t *out, unsigned int *olen, void *wmem)
{
	struct sw842_param *p = (struct sw842_param *)wmem;
	int ret;
	uint64_t last, next, pad, total;
	uint8_t repeat_count = 0;

	p->htable2 = NULL;
	p->htable4 = NULL;
	p->htable8 = NULL;

	p->in = (uint8_t *)in;
	p->instart = p->in;
	p->ilen = ilen;
	p->out = out;
	p->olen = *olen;
	p->bit = 0;

	total = p->olen;

	*olen = 0;
	/* if using strict mode, we can only compress a multiple of 8 */
	if (ilen % 8) {
		fprintf(stderr, "Can only compress multiples of 8 bytes, but len is len %d (%% 8 = %d)\n", ilen, ilen % 8);
		return -EINVAL;
	}

	/* let's compress at least 8 bytes, mkay? */
	if (unlikely(ilen < 8))
		goto skip_comp;

	/* make initial 'last' different so we don't match the first time */
	last = ~get_unaligned64((uint64_t *)p->in);

	while (p->ilen > 7) {
		next = get_unaligned64((uint64_t *)p->in);

		/* must get the next data, as we need to update the hashtable
		 * entries with the new data every time
		 */
		get_next_data(p);

		/* we don't care about endianness in last or next;
		 * we're just comparing 8 bytes to another 8 bytes,
		 * they're both the same endianness
		 */
		if (next == last) {
			/* repeat count bits are 0-based, so we stop at +1 */
			if (++repeat_count <= REPEAT_BITS_MAX)
				goto repeat;
		}
		if (repeat_count) {
			ret = add_repeat_template(p, repeat_count);
			repeat_count = 0;
			if (next == last) /* reached max repeat bits */
				goto repeat;
		}

		if (next == 0)
			ret = add_zeros_template(p);
		else
			ret = process_next(p);

		if (ret)
			return ret;

repeat:
		last = next;
		update_hashtables(p);
		p->in += 8;
		p->ilen -= 8;
	}

	if (repeat_count) {
		ret = add_repeat_template(p, repeat_count);
		if (ret)
			return ret;
	}

skip_comp:
	if (p->ilen > 0) {
		ret = add_short_data_template(p, p->ilen);
		if (ret)
			return ret;

		p->in += p->ilen;
		p->ilen = 0;
	}

	ret = add_end_template(p);
	if (ret)
		return ret;

	/*
	 * crc(0:31) is appended to target data starting with the next
	 * bit after End of stream template.
	 * nx842 calculates CRC for data in big-endian format. So doing
	 * same here so that sw842 decompression can be used for both
	 * compressed data.
	 */
	//boost::crc_32_type  crc;
	//crc.process_bytes(in, ilen);
	//crc = crc32_be(0, in, ilen);
	//ret = add_bits(p, crc.checksum(), CRC_BITS);
	if (ret)
		return ret;

	if (p->bit) {
		p->out++;
		p->olen--;
		p->bit = 0;
	}

	/* pad compressed length to multiple of 8 */
	pad = (8 - ((total - p->olen) % 8)) % 8;
	if (pad) {
		if (pad > p->olen) /* we were so close! */
			return -ENOSPC;
		memset(p->out, 0, pad);
		p->out += pad;
		p->olen -= pad;
	}

	if (unlikely((total - p->olen) > UINT_MAX))
		return -ENOSPC;

	*olen = total - p->olen;

	//printf("Out: %lld bits (%f bytes)\n", outbits, (outbits / 8.0f));

	return 0;
}
