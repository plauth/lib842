#ifndef __HASH_HPP__
#define __HASH_HPP__

#include <stdint.h>
#include <stdio.h>

//#define DEBUG           (1)

#define BUFFER16_BITS   (8)
#define BUFFER32_BITS   (9)
#define BUFFER64_BITS   (8)

#define DICT16_BITS     (12)
#define DICT32_BITS     (13)
#define DICT64_BITS     (12)

#define PRIME16		(65521)
#define PRIME32		(2654435761U)
#define PRIME64		(11400714785074694791ULL)

#define NO_ENTRY        (-1)

#ifndef __MEMACCESS_H__
#define __MEMACCESS_H__

typedef union { uint16_t value16; uint32_t value32; uint64_t value64; } __attribute__((packed)) unalign;

static uint16_t read16(const void* ptr) { return ((const unalign*)ptr)->value16; }
static uint32_t read32(const void* ptr) { return ((const unalign*)ptr)->value32; }
static uint64_t read64(const void* ptr) { return ((const unalign*)ptr)->value64; }

static void write16(void* ptr, uint16_t value) { ((unalign*)ptr)->value16 = value; }
static void write32(void* ptr, uint32_t value) { ((unalign*)ptr)->value32 = value; }
static void write32(void* ptr, uint64_t value) { ((unalign*)ptr)->value64 = value; }

#endif

static inline void checkRange(uint64_t value, uint64_t maxRange) {
        #ifdef DEBUG
        if(value >= maxRange) {
                printf("result exceeds valid range (%d): %llu\n", 1 << BUFFER16_BITS, value);
        }
        #endif
}

template<typename V, typename H, uint8_t dictBits> static inline H hash(V value) {
	V result;
        switch(sizeof(V)) {
                case 2:
                        result = PRIME16 * value;
                        result >>= (16 - dictBits);
                        checkRange(result, 1 << dictBits);
                        break;
                case 4:
                        result = PRIME32 * value;
                        result >>= 32 - dictBits;
                        checkRange(result, 1 << dictBits);
                        break;
                case 8:
                        result = PRIME64 * value;
                        result >>= 64 - dictBits;
                        checkRange(result, 1 << dictBits);
                        break;
                default:
                        fprintf(stderr, "Invalid template parameter V for function hash(V value)\n");
        }

	return (H) result;
}

template<typename T> static inline void replace_hash(int16_t *hashTable, T *ringBuffer, uint16_t index, T newValue, uint64_t *collisions) {
        uint16_t oldValueHash, newValueHash;
        T oldValue = ringBuffer[index];
        switch(sizeof(T)) {
                case 2:
                        oldValueHash = hash<T, uint16_t, DICT16_BITS>(oldValue);
                        newValueHash = hash<T, uint16_t, DICT16_BITS>(newValue);
                        break;
                case 4:
                        oldValueHash = hash<T, uint16_t, DICT32_BITS>(oldValue);
                        newValueHash = hash<T, uint16_t, DICT32_BITS>(newValue);
                        break;
                case 8:
                        oldValueHash = hash<T, uint16_t, DICT64_BITS>(oldValue);
                        newValueHash = hash<T, uint16_t, DICT64_BITS>(newValue);
                        break;
                default:
                        fprintf(stderr, "Invalid template parameter T for function replace_hash(...)\n");
        }

        //invalidate hash table entry for old data element that is pushed out of the ring buffer 
        hashTable[oldValueHash] = NO_ENTRY;

        //update ring buffer with new data element
        ringBuffer[index] = newValue;

        //update hash table entry for new data element
        #ifdef DEBUG
        if (hashTable[newValueHash] == NO_ENTRY) {
                hashTable[newValueHash] = index;
        } else {
                if (hashTable[newValueHash] != index && ringBuffer[hashTable[newValueHash]] != newValue) 
                        (*collisions)++;
                hashTable[newValueHash] = index;  
        }
        #else
        hashTable[newValueHash] = index;
        #endif
}

#endif
