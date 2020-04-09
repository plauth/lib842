#ifndef _KERNELDEPS_H
#define _KERNELDEPS_H

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


struct __attribute__((__packed__)) unaligned_uint16_t { uint16_t x; };
struct __attribute__((__packed__)) unaligned_uint32_t { uint32_t x; };
struct __attribute__((__packed__)) unaligned_uint64_t { uint64_t x; };

static inline uint16_t get_unaligned16(const void *p)
{
	const struct unaligned_uint16_t *ptr = (const struct unaligned_uint16_t *)p;
	return ptr->x;
}

static inline uint32_t get_unaligned32(const void *p)
{
	const struct unaligned_uint32_t *ptr = (const struct unaligned_uint32_t *)p;
	return ptr->x;
}

static inline uint64_t get_unaligned64(const void *p)
{
	const struct unaligned_uint64_t *ptr = (const struct unaligned_uint64_t *)p;
	return ptr->x;
}

static inline void put_unaligned_le16(uint16_t val, void *p)
{
	struct unaligned_uint16_t *ptr = (struct unaligned_uint16_t *)p;
	ptr->x = val;
}

static inline void put_unaligned_le32(uint32_t val, void *p)
{
	struct unaligned_uint32_t *ptr = (struct unaligned_uint32_t *)p;
	ptr->x = val;
}

static inline void put_unaligned_le64(uint64_t val, void *p)
{
	struct unaligned_uint64_t *ptr = (struct unaligned_uint64_t *)p;
	ptr->x = val;
}

#endif /* _KERNELDEPS_H */

