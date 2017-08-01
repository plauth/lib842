struct __una_uint16_t { uint16_t x; };
struct __una_uint32_t { uint32_t x; };
struct __una_uint64_t { uint64_t x; };

static inline uint16_t __get_unaligned_cpu16(const void *p)
{
	const struct __una_uint16_t *ptr = (const struct __una_uint16_t *)p;
	return ptr->x;
}

static inline uint32_t __get_unaligned_cpu32(const void *p)
{
	const struct __una_uint32_t *ptr = (const struct __una_uint32_t *)p;
	return ptr->x;
}

static inline uint64_t __get_unaligned_cpu64(const void *p)
{
	const struct __una_uint64_t *ptr = (const struct __una_uint64_t *)p;
	return ptr->x;
}

static inline void __put_unaligned_cpu16(uint16_t val, void *p)
{
	struct __una_uint16_t *ptr = (struct __una_uint16_t *)p;
	ptr->x = val;
}

static inline void __put_unaligned_cpu32(uint32_t val, void *p)
{
	struct __una_uint32_t *ptr = (struct __una_uint32_t *)p;
	ptr->x = val;
}

static inline void __put_unaligned_cpu64(uint64_t val, void *p)
{
	struct __una_uint64_t *ptr = (struct __una_uint64_t *)p;
	ptr->x = val;
}
