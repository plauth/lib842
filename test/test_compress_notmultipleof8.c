// Tests that compressing an amount of data that is not a multiple of 8 fails
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include "test_util.h"

static uint8_t NOT_MULTIPLE_OF_8[] = { 0x11, 0x11, 0x11, 0x12, 0x12 };

int main(int argc, char *argv[])
{
	const struct lib842_implementation *impl;
	if (argc != 2 || (impl = test842_get_impl_by_name(argv[1])) == NULL) {
		printf("test_compress_notmultipleof8 IMPL\n");
		return EXIT_FAILURE;
	}

	uint8_t *in = aligned_alloc(impl->required_alignment, sizeof(NOT_MULTIPLE_OF_8));
	memcpy(in, NOT_MULTIPLE_OF_8, sizeof(NOT_MULTIPLE_OF_8));
	size_t olen = 128;
	uint8_t *out = aligned_alloc(impl->required_alignment, olen);
	int ret = impl->compress(in, sizeof(NOT_MULTIPLE_OF_8), out, &olen);

	// This is a tricky one: The actual hardware can't compress input streams
	// that are not multiples of 8, but software implementations can, depending
	// on a flag. So simply allow both behaviors in the test.
	if (ret != 0 && ret != -EINVAL) {
		printf("Compression should have either succeeded or failed with EINVAL\n");
		return EXIT_FAILURE;
	}


	// Observation: On the real hardware hardware driver, the output of this
	// test gets added a special header (see the defintiions of
	// nx842_crypto_header and NX842_CRYPTO_MAGIC) on nx-842.{c,h}:
	// https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/crypto/nx/nx-842.c?id=c942fddf8793b2013be8c901b47d0a8dc02bf99f
	// https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/crypto/nx/nx-842.h?id=c942fddf8793b2013be8c901b47d0a8dc02bf99f
	// This appears to be intentional, but it means that the generated
	// output is not compatible with other decompressor implementations
	// (Since we don't validate this, it doesn't matter)

	// Additionally, also on the real hardware driver, there's the oddity
	// that even though after successful compression, olen is set to 31
	// (at least on my setup), passing an output buffer of size 32 or even
	// 64 will return ENOSPC, so it wants more space than strictly necessary

	return EXIT_SUCCESS;
}
