#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

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

typedef union { uint16_t value16; uint32_t value32; uint64_t value64; } __attribute__((packed)) unalign;

static uint16_t read16(const void* ptr) { return ((const unalign*)ptr)->value16; }
static uint32_t read32(const void* ptr) { return ((const unalign*)ptr)->value32; }
static uint64_t read64(const void* ptr) { return ((const unalign*)ptr)->value64; }

uint64_t nextMultipleOfEight(uint64_t input) {
	return (input + 7) & ~7;
} 

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

int main(int argc, char **argv) {
	if(argc <= 1) {
                printf("Wrong usage!\n");
	} else {
                printf("Memory consumption by hash tables: %lu KiB\n\n", ((1 << DICT16_BITS) * sizeof(int16_t) + (1 << DICT32_BITS) * sizeof(int16_t) + (1 << DICT64_BITS) * sizeof(int16_t)) / 1024);

		for(int i_args = 1; i_args < argc; i_args++) {
			FILE *fp;
			fp=fopen(argv[i_args], "r");
			fseek(fp, 0, SEEK_END);
			unsigned int flen = ftell(fp);
			fseek(fp, 0, SEEK_SET);
			unsigned int blen = nextMultipleOfEight(flen);
			printf("Computing all hash values for file '%s' (padded length: %d)\n", argv[i_args], blen);
			uint8_t *buffer = (uint8_t *) malloc(blen);
			memset(buffer, 0, blen);
			if(!fread(buffer, flen, 1, fp)) {
				fprintf(stderr, "FAIL: Reading file content to memory failed.\n");
			}
			fclose(fp);

			

                        int16_t  hashTable16[1 << DICT16_BITS], hashTable32[1 << DICT32_BITS], hashTable64[1 << DICT64_BITS];
                        uint16_t ringBuffer16[1 << BUFFER16_BITS];
			for(uint16_t i = 0; i < (1 << DICT16_BITS); i++) {
				hashTable16[i] = NO_ENTRY;
			}
			memset(ringBuffer16, 0, (1 << BUFFER16_BITS) * sizeof(uint16_t));
                        uint64_t collisions16 = 0;

                        //-----

                        uint32_t ringBuffer32[1 << BUFFER32_BITS];
                        for(uint16_t i = 0; i < (1 << DICT32_BITS); i++) {
                                hashTable32[i] = NO_ENTRY;
                        }
                        memset(ringBuffer32, 0, (1 << BUFFER32_BITS) * sizeof(uint32_t));
                        uint64_t collisions32 = 0;

                        //-----

                        uint64_t ringBuffer64[1 << BUFFER64_BITS];
                        for(uint16_t i = 0; i < (1 << DICT64_BITS); i++) {
                                hashTable64[i] = NO_ENTRY;
                        }
                        memset(ringBuffer64, 0, (1 << BUFFER64_BITS) *sizeof(uint32_t));
                        uint64_t collisions64 = 0;
			

			uint8_t *p = buffer;

                        while(p < buffer + blen){
                                uint64_t pos = p - buffer;
                                uint16_t i64 = (pos >> 3) % (1 << BUFFER64_BITS);
                                uint16_t i32 = (pos >> 2) % (1 << BUFFER32_BITS);
                                uint16_t i16 = (pos >> 1) % (1 << BUFFER16_BITS);

                                replace_hash<uint16_t>(hashTable16, ringBuffer16, i16  , read16(p  ), &collisions16);
                                replace_hash<uint16_t>(hashTable16, ringBuffer16, i16+1, read16(p+2), &collisions16);
                                replace_hash<uint16_t>(hashTable16, ringBuffer16, i16+2, read16(p+4), &collisions16);
                                replace_hash<uint16_t>(hashTable16, ringBuffer16, i16+3, read16(p+6), &collisions16);

                                replace_hash<uint32_t>(hashTable32, ringBuffer32, i32  , read32(p  ), &collisions32);
                                replace_hash<uint32_t>(hashTable32, ringBuffer32, i32+1, read32(p+4), &collisions32);

                                replace_hash<uint64_t>(hashTable64, ringBuffer64, i64  , read64(p  ), &collisions64);


                                p += 8;
                        }

                        #ifdef DEBUG
                        printf("Collisions for 2 byte values: %llu (%f%%)\n", collisions16, ((float) collisions16 / (float) (blen >> 1) * 100));
                        printf("Collisions for 4 byte values: %llu (%f%%)\n", collisions32, ((float) collisions32 / (float) (blen >> 2) * 100));
                        printf("Collisions for 8 byte values: %llu (%f%%)\n", collisions64, ((float) collisions64 / (float) (blen >> 3) * 100));
			printf("\n\n");
                        #endif

			free(buffer);
		}
	}
}
