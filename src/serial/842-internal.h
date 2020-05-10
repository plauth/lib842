#ifndef LIB842_SRC_SERIAL_842_INTERNAL_H
#define LIB842_SRC_SERIAL_842_INTERNAL_H

#include "../common/842.h"
#include "uthash/src/uthash.h"
#include "uthash/src/utlist.h"

#include <stdint.h>

//#define DEBUG 1

struct node2_el {
	uint8_t index;
	struct node2_el *next, *prev;
};

struct hlist_node2 {
	uint16_t data;
	struct node2_el *head;
	UT_hash_handle hh;
};

struct node4_el {
	uint16_t index;
	struct node4_el *next, *prev;
};

struct hlist_node4 {
	uint32_t data;
	struct node4_el *head;
	UT_hash_handle hh;
};

struct node8_el {
	uint8_t index;
	struct node8_el *next, *prev;
};

struct hlist_node8 {
	uint64_t data;
	struct node8_el *head;
	UT_hash_handle hh;
};

struct sw842_param {
	const uint8_t *in;
	const uint8_t *instart;
	uint64_t ilen;
	uint8_t *out;
	uint64_t olen;
	uint8_t bit;
	uint64_t data8[1];
	uint32_t data4[2];
	uint16_t data2[4];
	int index8[1];
	int index4[2];
	int index2[4];
	struct hlist_node8 *htable8;
	struct hlist_node4 *htable4;
	struct hlist_node2 *htable2;
	uint64_t node8[1 << I8_BITS];
	uint32_t node4[1 << I4_BITS];
	uint16_t node2[1 << I2_BITS];
};

struct sw842_param_decomp {
	const uint8_t *in;
	uint8_t bit;
	uint64_t ilen;
	uint8_t *out;
	const uint8_t *ostart;
	uint64_t olen;
};

#endif // LIB842_SRC_SERIAL_842_INTERNAL_H
