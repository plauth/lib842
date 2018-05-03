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
#include "../common/opcodes.h"

//static uint64_t outbits = 0;

#define INDEX_NOT_FOUND		(-1)
#define INDEX_NOT_CHECKED	(-2)

/*
template<typename V, uint8_t dictBits> static inline void hash_vec(V* __restrict__ input, V* __restrict__ hash) {
        switch(sizeof(V)) {
                case 2:
                		for(int i = 0; i < 4; i++) {
                                hash[i] = input[i] * PRIME16;
                        }
                        for(int i = 0; i < 4; i++) {
                                hash[i] >>= (16 - dictBits);
                        }
                        break;
                case 4:
                        hash[0] = input[0] * PRIME32;
                        hash[1] = input[1] * PRIME32;
                        
                        hash[0] >>= 32 - dictBits;
                        hash[1] >>= 32 - dictBits;
                        break;
                case 8:
                        hash[0] = input[0] * PRIME64;
                        hash[0] >>= 64 - dictBits;
                        break;
                default:
                        fprintf(stderr, "Invalid template parameter V for function hash(V input)\n");
        }
}*/

template<typename T> static inline void replace_hash(struct sw842_param *p, uint16_t index, uint16_t offset) {
		uint16_t ringBufferIndex = index + offset;
        uint16_t oldValueHash, newValueHash;

        switch(sizeof(T)) {
                case 2:
                		//hash_vec<T, DICT16_BITS>((T*)p->data2, (T*)p->hash2);
                        oldValueHash = hash<T, uint16_t, DICT16_BITS>(p->ringBuffer16[ringBufferIndex]);
                        newValueHash = hash<T, uint16_t, DICT16_BITS>(p->data2[offset]);
                        p->hashTable16[oldValueHash] = NO_ENTRY;
                        p->ringBuffer16[ringBufferIndex] = p->data2[offset];
                        p->hashTable16[newValueHash] = ringBufferIndex;
                        break;
                case 4:
                        oldValueHash = hash<T, uint16_t, DICT32_BITS>(p->ringBuffer32[ringBufferIndex]);
                        newValueHash = hash<T, uint16_t, DICT32_BITS>(p->data4[offset]);
                        p->hashTable32[oldValueHash] = NO_ENTRY;
                        p->ringBuffer32[ringBufferIndex] = p->data4[offset];
                        p->hashTable32[newValueHash] = ringBufferIndex;
                        break;
                case 8:
                        oldValueHash = hash<T, uint16_t, DICT64_BITS>(p->ringBuffer64[ringBufferIndex]);
                        newValueHash = hash<T, uint16_t, DICT64_BITS>(p->data4[offset]);
                        p->hashTable64[oldValueHash] = NO_ENTRY;
                        p->ringBuffer64[ringBufferIndex] = p->data8[offset];
                        p->hashTable64[newValueHash] = ringBufferIndex;
                        break;
                default:
                        fprintf(stderr, "Invalid template parameter T for function replace_hash(...)\n");
        }
}

template<typename T> static inline void find_index(struct sw842_param *p, uint16_t offset) {
		uint16_t hashValue;
		int16_t index;
        switch(sizeof(T)) {
                case 2:
                	p->index2[offset] = INDEX_NOT_FOUND;
                	hashValue = hash<T, uint16_t, DICT16_BITS>(p->data2[offset]);
                	index = p->hashTable16[hashValue];
                	if(index != NO_ENTRY && p->ringBuffer16[index] == p->data2[offset]) {
                		p->index2[offset] = index;
                	}
                	break;
                case 4:
                	p->index4[offset] = INDEX_NOT_FOUND;
                	hashValue = hash<T, uint16_t, DICT32_BITS>(p->data4[offset]);
                	index = p->hashTable32[hashValue];
                	if(index != NO_ENTRY && p->ringBuffer32[index] == p->data4[offset]) {
                		p->index4[offset] = index;
                	}
                	break;
                case 8:
                	p->index8[offset] = INDEX_NOT_FOUND;
                	hashValue = hash<T, uint16_t, DICT64_BITS>(p->data8[offset]);
                	index = p->hashTable64[hashValue];
                	if(index != NO_ENTRY && p->ringBuffer64[index] == p->data8[offset]) {
                		p->index8[offset] = index;
                	}
                	break;
        }
}

static inline uint8_t get_template(struct sw842_param *p) {
		uint16_t template_key = 0;
		
		if(p->index8[0] >= 0) {
			template_key = I8 * 3;
		} else {
			if(p->index4[0] >= 0) {
				template_key += I4 * 3;
			} else {
				if(p->index2[0] >= 0)
					template_key += I2 * 3;
				if(p->index2[1] >= 0)
					template_key += I2 * 5;
			}
			if(p->index4[1] >= 0) {
				template_key += I4 * 5;
			} else {
				if(p->index2[2] >= 0)
					template_key += I2 * 7;
				if(p->index2[3] >= 0)
					template_key += I2 * 11;
			}			
		}

		template_key >>= 2;



		if (template_key > 117) {
			fprintf(stderr, "Invalid template_key '%d', the key is larger than maximum value 117!\n", template_key);
			exit(-EINVAL);
		}
		
		return ops_dict[template_key];
}

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
	int b = p->bit;
	int bits = b + n;
	int s = round_up(bits, 8) - bits;
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

	if (bytes_rounded_up(bits) > p->olen)
		return -ENOSPC;
 
	//outbits += n;
	o = *out & bmask[b];
	d <<= s;

	if (bits <= 8)
		*out = o | d;
	else if (bits <= 16)
		write16(out, swap_endianness16(o << 8 | d));
	else if (bits <= 24)
		write32(out, swap_endianness32(o << 24 | d << 8));
	else if (bits <= 32)
		write32(out, swap_endianness32(o << 24 | d));
	else if (bits <= 40)
		write64(out, swap_endianness64(o << 56 | d << 24));
	else if (bits <= 48)
		write64(out, swap_endianness64(o << 56 | d << 16));
	else if (bits <= 56)
		write64(out, swap_endianness64(o << 56 | d << 8));
	else
		write64(out, swap_endianness64(o << 56 | d));

	p->bit += n;

	if (p->bit > 7) {
		p->out += p->bit / 8;
		p->olen -= p->bit / 8;
		p->bit %= 8;
	}

	return 0;
}

template<uint8_t TEMPLATE_KEY> static inline int add_template(struct sw842_param *p) {
	int ret = 0;
	ret = add_bits(p, TEMPLATE_KEY, OP_BITS);
	if (ret)
		return ret;

    switch(TEMPLATE_KEY) {
        case 0x00: 	// { D8, N0, N0, N0 }, 64 bits
        	ret = add_bits(p, p->data8[0], 64);
    	    break;
        case 0x01:	// { D4, D2, I2, N0 }, 56 bits
        	ret |= add_bits(p, p->data4[0], 32);
        	ret |= add_bits(p, p->data2[2], 16);
        	ret |= add_bits(p, p->index2[3], I2_BITS);
    	    break;
        case 0x02:	// { D4, I2, D2, N0 }, 56 bits
        	ret |= add_bits(p, p->data4[0], 32);
			ret |= add_bits(p, p->index2[2], I2_BITS);
			ret |= add_bits(p, p->data2[3], 16);
    	    break;
		case 0x03: 	// { D4, I2, I2, N0 }, 48 bits
			ret |= add_bits(p, p->data4[0], 32);
    		ret |= add_bits(p, p->index2[2], I2_BITS);
    		ret |= add_bits(p, p->index2[3], I2_BITS);
    	    break;
		case 0x04:	// { D4, I4, N0, N0 }, 41 bits
			ret |= add_bits(p, p->data4[0], 32);
			ret |= add_bits(p, p->index4[1], I4_BITS);
    	    break;
		case 0x05:	// { D2, I2, D4, N0 }, 56 bits
			ret |= add_bits(p, p->data2[0], 16);
			ret |= add_bits(p, p->index2[1], I2_BITS);
			ret |= add_bits(p, p->data4[1], 32);
    	    break;
		case 0x06:	// { D2, I2, D2, I2 }, 48 bits
			ret |= add_bits(p, p->data2[0], 16);
			ret |= add_bits(p, p->index2[1], I2_BITS);
			ret |= add_bits(p, p->data2[2], 16);
			ret |= add_bits(p, p->index2[3], I2_BITS);
    	    break;
		case 0x07:	// { D2, I2, I2, D2 }, 48 bits
			ret |= add_bits(p, p->data2[0], 16);
			ret |= add_bits(p, p->index2[1], I2_BITS);
			ret |= add_bits(p, p->index2[2], I2_BITS);
			ret |= add_bits(p, p->data2[3], 16);
    	    break;
		case 0x08:	// { D2, I2, I2, I2 }, 40 bits
			ret |= add_bits(p, p->data2[0], 16);
			ret |= add_bits(p, p->index2[1], I2_BITS);
			ret |= add_bits(p, p->index2[2], I2_BITS);
			ret |= add_bits(p, p->index2[3], I2_BITS);
    	    break;
		case 0x09:	// { D2, I2, I4, N0 }, 33 bits
			ret |= add_bits(p, p->data2[0], 16);
			ret |= add_bits(p, p->index2[1], I2_BITS);
			ret |= add_bits(p, p->index4[1], I4_BITS);
    	    break;
		case 0x0a:	// { I2, D2, D4, N0 }, 56 bits
    		ret |= add_bits(p, p->index2[0], I2_BITS);
    		ret |= add_bits(p, p->data2[1], 16);
    		ret |= add_bits(p, p->data4[1], 32);
    	    break;
		case 0x0b:	// { I2, D4, I2, N0 }, 48 bits
			ret |= add_bits(p, p->index2[0], I2_BITS);
			ret |= add_bits(p, swap_endianness32(read32(p->in + 2)), 32);
			ret |= add_bits(p, p->index2[3], I2_BITS);
    	    break;
		case 0x0c:	// { I2, D2, I2, D2 }, 48 bits
			ret |= add_bits(p, p->index2[0], I2_BITS);
    		ret |= add_bits(p, p->data2[1], 16);
			ret |= add_bits(p, p->index2[2], I2_BITS);
    		ret |= add_bits(p, p->data2[3], 16);
    	    break;
		case 0x0d:	// { I2, D2, I2, I2 }, 40 bits
			ret |= add_bits(p, p->index2[0], I2_BITS);
    		ret |= add_bits(p, p->data2[1], 16);
			ret |= add_bits(p, p->index2[2], I2_BITS);
    		ret |= add_bits(p, p->index2[3], I2_BITS);
    	    break;
		case 0x0e:	// { I2, D2, I4, N0 }, 33 bits
			ret |= add_bits(p, p->index2[0], I2_BITS);
    		ret |= add_bits(p, p->data2[1], 16);
    		ret |= add_bits(p, p->index4[1], I4_BITS);
    	    break;
		case 0x0f:	// { I2, I2, D4, N0 }, 48 bits
    		ret |= add_bits(p, p->index2[0], I2_BITS);
    		ret |= add_bits(p, p->index2[1], I2_BITS);
    		ret |= add_bits(p, p->data4[1], 32);
    	    break;
		case 0x10:	// { I2, I2, D2, I2 }, 40 bits
    		ret |= add_bits(p, p->index2[0], I2_BITS);
    		ret |= add_bits(p, p->index2[1], I2_BITS);
    		ret |= add_bits(p, p->data2[2], 16);
       		ret |= add_bits(p, p->index2[3], I2_BITS);
    	    break;
		case 0x11:	// { I2, I2, I2, D2 }, 40 bits
    		ret |= add_bits(p, p->index2[0], I2_BITS);
    		ret |= add_bits(p, p->index2[1], I2_BITS);
       		ret |= add_bits(p, p->index2[2], I2_BITS);
    		ret |= add_bits(p, p->data2[3], 16);
    	    break;
		case 0x12:	// { I2, I2, I2, I2 }, 32 bits
    		ret |= add_bits(p, p->index2[0], I2_BITS);
    		ret |= add_bits(p, p->index2[1], I2_BITS);
       		ret |= add_bits(p, p->index2[2], I2_BITS);
       		ret |= add_bits(p, p->index2[3], I2_BITS);
    	    break;
		case 0x13:	// { I2, I2, I4, N0 }, 25 bits
    		ret |= add_bits(p, p->index2[0], I2_BITS);
    		ret |= add_bits(p, p->index2[1], I2_BITS);
       		ret |= add_bits(p, p->index4[1], I4_BITS);
    	    break;
		case 0x14:	// { I4, D4, N0, N0 }, 41 bits
       		ret |= add_bits(p, p->index4[0], I4_BITS);
    		ret |= add_bits(p, p->data4[1], 32);
    	    break;
		case 0x15:	// { I4, D2, I2, N0 }, 33 bits
       		ret |= add_bits(p, p->index4[0], I4_BITS);
    		ret |= add_bits(p, p->data2[2], 16);
    		ret |= add_bits(p, p->index2[3], I2_BITS);
    	    break;
		case 0x16:	// { I4, I2, D2, N0 }, 33 bits
       		ret |= add_bits(p, p->index4[0], I4_BITS);
    		ret |= add_bits(p, p->index2[2], I2_BITS);
    		ret |= add_bits(p, p->data2[3], 16);
    	    break;
		case 0x17:	// { I4, I2, I2, N0 }, 25 bits
       		ret |= add_bits(p, p->index4[0], I4_BITS);
    		ret |= add_bits(p, p->index2[2], I2_BITS);
    		ret |= add_bits(p, p->index2[3], I2_BITS);
    	    break;
		case 0x18:	// { I4, I4, N0, N0 }, 18 bits
    		ret |= add_bits(p, p->index4[0], I4_BITS);
			ret |= add_bits(p, p->index4[1], I4_BITS);
    	    break;
		case 0x19:	// { I8, N0, N0, N0 }, 8 bits
    		ret = add_bits(p, p->index8[0], I8_BITS);
    	    break;
        default:
        	fprintf(stderr, "Invalid template: %x\n", TEMPLATE_KEY);
        }

		if (ret)
			return ret;

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

static void get_next_data(struct sw842_param *p)
{
	p->data8[0] = swap_endianness64(read64(p->in    ));
	p->data4[0] = swap_endianness32(read32(p->in    ));
	p->data4[1] = swap_endianness32(read32(p->in + 4));
	p->data2[0] = swap_endianness16(read16(p->in    ));
	p->data2[1] = swap_endianness16(read16(p->in + 2));
	p->data2[2] = swap_endianness16(read16(p->in + 4));
	p->data2[3] = swap_endianness16(read16(p->in + 6));
}

/* update the hashtable entries.
 * only call this after finding/adding the current template
 * the dataN fields for the current 8 byte block must be already updated
 */
static void update_hashtables(struct sw842_param *p)
{

	uint64_t pos = p->in - p->instart;
    uint16_t i64 = (pos >> 3) % (1 << BUFFER64_BITS);
    uint16_t i32 = (pos >> 2) % (1 << BUFFER32_BITS);
    uint16_t i16 = (pos >> 1) % (1 << BUFFER16_BITS);

    replace_hash<uint16_t>(p, i16, 0);
    replace_hash<uint16_t>(p, i16, 1);
    replace_hash<uint16_t>(p, i16, 2);
    replace_hash<uint16_t>(p, i16, 3);
    replace_hash<uint32_t>(p, i32, 0);
    replace_hash<uint32_t>(p, i32, 1);
    replace_hash<uint64_t>(p, i64, 0);
}

/* find the next template to use, and add it
 * the p->dataN fields must already be set for the current 8 byte block
 */
static int process_next(struct sw842_param *p)
{
	int ret;
	uint8_t template_key;

	p->index8[0] = INDEX_NOT_CHECKED;
	p->index4[0] = INDEX_NOT_CHECKED;
	p->index4[1] = INDEX_NOT_CHECKED;
	p->index2[0] = INDEX_NOT_CHECKED;
	p->index2[1] = INDEX_NOT_CHECKED;
	p->index2[2] = INDEX_NOT_CHECKED;
	p->index2[3] = INDEX_NOT_CHECKED;

	find_index<uint64_t>(p, 0);
	find_index<uint32_t>(p, 0);
	find_index<uint32_t>(p, 1);
	find_index<uint16_t>(p, 0);
	find_index<uint16_t>(p, 1);
	find_index<uint16_t>(p, 2);
	find_index<uint16_t>(p, 3);

	template_key = get_template(p);

	switch(template_key) {
        case 0x00: 	// { D8, N0, N0, N0 }, 64 bits
    		ret = add_template<0x00>(p);
    	    break;
        case 0x01:	// { D4, D2, I2, N0 }, 56 bits
    		ret = add_template<0x01>(p);
    	    break;
        case 0x02:	// { D4, I2, D2, N0 }, 56 bits
    		ret = add_template<0x02>(p);
    	    break;
		case 0x03: 	// { D4, I2, I2, N0 }, 48 bits
    		ret = add_template<0x03>(p);
    	    break;
		case 0x04:	// { D4, I4, N0, N0 }, 41 bits
    		ret = add_template<0x04>(p);
    	    break;
		case 0x05:	// { D2, I2, D4, N0 }, 56 bits
    		ret = add_template<0x05>(p);
    	    break;
		case 0x06:	// { D2, I2, D2, I2 }, 48 bits
    		ret = add_template<0x06>(p);
    	    break;
		case 0x07:	// { D2, I2, I2, D2 }, 48 bits
    		ret = add_template<0x07>(p);
    	    break;
		case 0x08:	// { D2, I2, I2, I2 }, 40 bits
    		ret = add_template<0x08>(p);
    	    break;
		case 0x09:	// { D2, I2, I4, N0 }, 33 bits
    		ret = add_template<0x09>(p);
    	    break;
		case 0x0a:	// { I2, D2, D4, N0 }, 56 bits
    		ret = add_template<0x0a>(p);
    	    break;
		case 0x0b:	// { I2, D4, I2, N0 }, 48 bits
    		ret = add_template<0x0b>(p);
    	    break;
		case 0x0c:	// { I2, D2, I2, D2 }, 48 bits
    		ret = add_template<0x0c>(p);
    	    break;
		case 0x0d:	// { I2, D2, I2, I2 }, 40 bits
    		ret = add_template<0x0d>(p);
    	    break;
		case 0x0e:	// { I2, D2, I4, N0 }, 33 bits
    		ret = add_template<0x0e>(p);
    	    break;
		case 0x0f:	// { I2, I2, D4, N0 }, 48 bits
    		ret = add_template<0x0f>(p);
    	    break;
		case 0x10:	// { I2, I2, D2, I2 }, 40 bits
    		ret = add_template<0x10>(p);
    	    break;
		case 0x11:	// { I2, I2, I2, D2 }, 40 bits
    		ret = add_template<0x11>(p);
    	    break;
		case 0x12:	// { I2, I2, I2, I2 }, 32 bits
    		ret = add_template<0x12>(p);
    	    break;
		case 0x13:	// { I2, I2, I4, N0 }, 25 bits
    		ret = add_template<0x13>(p);
    	    break;
		case 0x14:	// { I4, D4, N0, N0 }, 41 bits
    		ret = add_template<0x14>(p);
    	    break;
		case 0x15:	// { I4, D2, I2, N0 }, 33 bits
    		ret = add_template<0x15>(p);
    	    break;
		case 0x16:	// { I4, I2, D2, N0 }, 33 bits
    		ret = add_template<0x16>(p);
    	    break;
		case 0x17:	// { I4, I2, I2, N0 }, 25 bits
    		ret = add_template<0x17>(p);
    	    break;
		case 0x18:	// { I4, I4, N0, N0 }, 18 bits
    		ret = add_template<0x18>(p);
    	    break;
		case 0x19:	// { I8, N0, N0, N0 }, 8 bits
    		ret = add_template<0x19>(p);
    	    break;
        default:
        	fprintf(stderr, "Invalid template: %x\n", template_key);
        }

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
		   uint8_t *out, unsigned int *olen)
{
	struct sw842_param *p = (struct sw842_param *) malloc(sizeof(struct sw842_param)); 

	int ret;
	uint64_t last, next, pad, total;
	uint8_t repeat_count = 0;
	uint32_t crc;

	for(uint16_t i = 0; i < (1 << DICT16_BITS); i++) {
		p->hashTable16[i] = NO_ENTRY;
	}

    for(uint16_t i = 0; i < (1 << DICT32_BITS); i++) {
            p->hashTable32[i] = NO_ENTRY;
    }

    for(uint16_t i = 0; i < (1 << DICT64_BITS); i++) {
            p->hashTable64[i] = NO_ENTRY;
    }

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

	/* make initial 'last' different so we don't match the first time */
	last = ~read64((uint64_t *)p->in);

	while (p->ilen > 7) {
		next = read64((uint64_t *)p->in);

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
	crc = crc32_be(0, (const unsigned char *) in, ilen);

	ret = add_bits(p, crc, CRC_BITS);
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

	if ((total - p->olen) > UINT_MAX)
		return -ENOSPC;

	*olen = total - p->olen;

	//printf("Out: %lld bits (%f bytes)\n", outbits, (outbits / 8.0f));
	free(p);

	return 0;
}
