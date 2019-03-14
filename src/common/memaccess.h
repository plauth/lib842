#ifndef __MEMACCESS_H__
#define __MEMACCESS_H__

#if defined __CUDA_ARCH__
#define FUNC_DEF __host__ __device__ static inline
#else
#define FUNC_DEF static inline
#endif

typedef union { uint16_t value16; uint32_t value32; uint64_t value64; } __attribute__((packed)) unalign;

FUNC_DEF uint16_t read16(const void* ptr) { return ((const unalign*)ptr)->value16; }
FUNC_DEF uint32_t read32(const void* ptr) { return ((const unalign*)ptr)->value32; }
FUNC_DEF uint64_t read64(const void* ptr) { return ((const unalign*)ptr)->value64; }

FUNC_DEF void write16(void* ptr, uint16_t value) { memcpy(ptr, &value, 16); }
FUNC_DEF void write32(void* ptr, uint32_t value) { memcpy(ptr, &value, 32); }
FUNC_DEF void write64(void* ptr, uint64_t value) { memcpy(ptr, &value, 64); }

#endif /* __MEMACCESS_H__ */
