typedef uint16_t __be16;
typedef uint32_t __be32;
typedef uint64_t __be64;

#define BITS_PER_LONG_LONG 64
#define GENMASK_ULL(h, l) \
	(((~0ULL) << (l)) & (~0ULL >> (BITS_PER_LONG_LONG - 1 - (h))))
		
#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define round_up(x, y) ((((x)-1) | __round_mask(x, y))+1)
#define round_down(x, y) ((x) & ~__round_mask(x, y))
	
#define DIV_ROUND_UP(n,d) (((n) + (d) - 1) / (d))
		
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)
		
