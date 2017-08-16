
static inline uint16_t swap_endianness16(uint16_t input) {
	#ifdef __x86_64__
		asm("xchgb %b0,%h0" : "=Q" (input) :  "0" (input));
		return input;
	//#elif defined __ppc64__
	//#elif defined __CUDA_ARCH__
	#else
		uint16_t b0,b1;

		b0 = (input & 0x00ff) << 8u;
		b1 = (input & 0xff00) >> 8u;

		return b0 | b1;
	#endif

}

static inline uint32_t swap_endianness32(uint32_t input) {
	#ifdef __x86_64__
		asm("bswap %0" : "=r" (input) : "0" (input));
		return input;
	//#elif defined __ppc64__
	//#elif defined __CUDA_ARCH__
	#else
		uint32_t b0,b1,b2,b3;

		b0 = (input & 0x000000ff) << 24u;
		b1 = (input & 0x0000ff00) <<  8u;
		b2 = (input & 0x00ff0000) >>  8u;
		b3 = (input & 0xff000000) >> 24u;

		return b0 | b1 | b2 | b3;
	#endif
}

static inline uint64_t swap_endianness64(uint64_t input) {
	#ifdef __x86_64__
		asm("bswap %0" : "=r" (input) : "0" (input));
		return input;
	//#elif defined __ppc64__
	//#elif defined __CUDA_ARCH__
	#else
		uint64_t b0,b1,b2,b3,b4,b5,b6,b7;

		b0 = (input & 0x00000000000000ff) << 56u;
		b1 = (input & 0x000000000000ff00) << 40u;
		b2 = (input & 0x0000000000ff0000) << 24u;
		b3 = (input & 0x00000000ff000000) <<  8u;
		b4 = (input & 0x000000ff00000000) >>  8u;
		b5 = (input & 0x0000ff0000000000) >> 24u;
		b6 = (input & 0x00ff000000000000) >> 40u;
		b7 = (input & 0xff00000000000000) >> 56u;

		return b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7;
	#endif
}
