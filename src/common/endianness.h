#ifndef __ENDIANNESS_H__
#define __ENDIANNESS_H__

#if defined __CUDA_ARCH__
#define FUNC_DEF __host__ __device__ static inline
#else
#define FUNC_DEF static inline
#endif

FUNC_DEF uint16_t swap_endianness16(uint16_t input) {
	#ifdef __x86_64__
		asm("xchgb %b0,%h0" : "=Q" (input) :  "0" (input));
		return input;
	//#elif defined __CUDA_ARCH__
	#else
		return
		((input & (uint16_t)0x00ffU) << 8) |
		((input & (uint16_t)0xff00U) >> 8);
	#endif

}

FUNC_DEF uint16_t swap_be_to_native16(uint16_t input){
	#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
		return swap_endianness16(input);
	#else
		return input;
	#endif
}

FUNC_DEF uint16_t swap_native_to_be16(uint16_t input){
	#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
		return swap_endianness16(input);
	#else
		return input;
	#endif
}

FUNC_DEF uint32_t swap_endianness32(uint32_t input) {
	#ifdef __x86_64__
		asm("bswap %0" : "=r" (input) : "0" (input));
		return input;
	//#elif defined __CUDA_ARCH__
	#else
		return
		((input & (uint32_t)0x000000ffUL) << 24) |
		((input & (uint32_t)0x0000ff00UL) <<  8) |
		((input & (uint32_t)0x00ff0000UL) >>  8) |
		((input & (uint32_t)0xff000000UL) >> 24);
	#endif
}

FUNC_DEF uint32_t swap_be_to_native32(uint32_t input){
	#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
		return swap_endianness32(input);
	#else
		return input;
	#endif
}

FUNC_DEF uint32_t swap_native_to_be32(uint32_t input){
	#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
		return swap_endianness32(input);
	#else
		return input;
	#endif
}

FUNC_DEF uint64_t swap_endianness64(uint64_t input) {
	#ifdef __x86_64__
		asm("bswap %0" : "=r" (input) : "0" (input));
		return input;
	//#elif defined __CUDA_ARCH__
	#else
		return
		(uint64_t)((input & (uint64_t)0x00000000000000ffULL) << 56) |
		(uint64_t)((input & (uint64_t)0x000000000000ff00ULL) << 40) |
		(uint64_t)((input & (uint64_t)0x0000000000ff0000ULL) << 24) |
		(uint64_t)((input & (uint64_t)0x00000000ff000000ULL) <<  8) |
		(uint64_t)((input & (uint64_t)0x000000ff00000000ULL) >>  8) |
		(uint64_t)((input & (uint64_t)0x0000ff0000000000ULL) >> 24) |
		(uint64_t)((input & (uint64_t)0x00ff000000000000ULL) >> 40) |
		(uint64_t)((input & (uint64_t)0xff00000000000000ULL) >> 56);
	#endif
}

FUNC_DEF uint64_t swap_be_to_native64(uint64_t input){
	#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
		return swap_endianness64(input);
	#else
		return input;
	#endif
}

FUNC_DEF uint64_t swap_native_to_be64(uint64_t input){
	#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
		return swap_endianness64(input);
	#else
		return input;
	#endif
}
#endif
