#ifndef __MEMACCESS_H__
#define __MEMACCESS_H__

typedef union { uint16_t value16; uint32_t value32; uint64_t value64; } __attribute__((packed)) unalign;

static inline uint16_t read16(const void* ptr) { return ((const unalign*)ptr)->value16; }
static inline uint32_t read32(const void* ptr) { return ((const unalign*)ptr)->value32; }
static inline uint64_t read64(const void* ptr) { return ((const unalign*)ptr)->value64; }

static inline void write16(void* ptr, uint16_t value) { ((unalign*)ptr)->value16 = value; }
static inline void write32(void* ptr, uint32_t value) { ((unalign*)ptr)->value32 = value; }
static inline void write64(void* ptr, uint64_t value) { ((unalign*)ptr)->value64 = value; }

#endif /* __MEMACCESS_H__ */
