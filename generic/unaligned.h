#ifndef _UNALIGNED_H
#define _UNALIGNED_H

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

#endif /* _UNALIGNED_H */
