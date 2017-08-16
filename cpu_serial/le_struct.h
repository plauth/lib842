#ifndef _LINUX_UNALIGNED_LE_STRUCT_H
#define _LINUX_UNALIGNED_LE_STRUCT_H

#include "packed_struct.h"

static inline uint16_t get_unaligned_le16(const void *p)
{
	return __get_unaligned_cpu16((const uint8_t *)p);
}

static inline uint32_t get_unaligned_le32(const void *p)
{
	return __get_unaligned_cpu32((const uint8_t *)p);
}

static inline uint64_t get_unaligned_le64(const void *p)
{
	return __get_unaligned_cpu64((const uint8_t *)p);
}

static inline void put_unaligned_le16(uint16_t val, void *p)
{
	__put_unaligned_cpu16(val, p);
}

static inline void put_unaligned_le32(uint32_t val, void *p)
{
	__put_unaligned_cpu32(val, p);
}

static inline void put_unaligned_le64(uint64_t val, void *p)
{
	__put_unaligned_cpu64(val, p);
}

#endif /* _LINUX_UNALIGNED_LE_STRUCT_H */
