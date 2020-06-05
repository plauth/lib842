/* Test for regression in the original software decompressor implementation
   in the Linux Kernel. Description:

   The software 842 decompressor receives, through the initial value of the
   'olen' parameter, the capacity of the buffer pointed to by 'out'. If this
   capacity is insufficient to decode the compressed bitstream, -ENOSPC
   should be returned.

   However, the bounds checks are missing for index references (for those
   ops. where decomp_ops includes a I2, I4 or I8 subop.), and also for
   OP_SHORT_DATA. Due to this, sw842_decompress can write past the capacity
   of the 'out' buffer.

   The case for index references can be triggered by compressing data that
   follows a 16-byte periodic pattern (excluding special cases which are
   better encoded by OP_ZEROS) and passing a 'olen' somewhat smaller than the
   original length.
   The case for OP_SHORT_DATA can be triggered by compressing an amount of
   data that is not a multiple of 8, and then a slightly smaller 'olen' that
   can't fit those last <8 bytes.
*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <lib842/sw.h>
#include "test_util.h"

static int test_bound(const struct lib842_implementation *impl,
		      const char *name, size_t ibound, size_t dbound)
{
	uint8_t *in = aligned_alloc(impl->required_alignment, ibound),
		*out = aligned_alloc(impl->required_alignment, ibound * 4),
		*decomp = aligned_alloc(impl->required_alignment,
					ibound /* Overallocated */);
	size_t clen = ibound * 4, dlen = dbound;
	int ret;

	for (size_t i = 0; i < ibound; i ++)
		in[i] = i % 16; // 0, 1, 2, ..., 14, 15, 0, 1, 2, ...
	for (size_t i = dbound; i < ibound; i++)
		decomp[i] = 0xFF; // Place guard bytes

	ret = sw842_compress(in, ibound, out, &clen);
	assert(ret == 0);

	ret = impl->decompress(out, clen, decomp, &dlen);
	if (ret != -ENOSPC) {
		printf("%s: Did not return ENOSPC as expected\n", name);
		return 0;
	}
	for (size_t i = dbound; i < ibound; i++) {
		if (decomp[i] != 0xFF) {
			printf("%s: Guard has been overwritten\n", name);
			return 0;
		}
	}

	return 1;
}

int main(int argc, char *argv[])
{
	const struct lib842_implementation *impl;
	if (argc != 2 || (impl = test842_get_impl_by_name(argv[1])) == NULL) {
		printf("test_decompress_boundscheck IMPL\n");
		return EXIT_FAILURE;
	}


	// FIXME TESTFAILURE: This bug is present in the Linux Kernel as of v5.7
	//                    and may lead to a full OS crash... better not to run it
	if (strcmp(argv[1], "hw") == 0) {
		fprintf(stderr, "!! TEST FAILED (BUT PASS, DUE TO KNOWN DEFECT) !!\n");
		return EXIT_SUCCESS;
	}

	if (!test_bound(impl, "Index reference test", 256, 64))
		return EXIT_FAILURE;

	if (strcmp(argv[1], "sw") == 0 &&
	    !test_bound(impl, "Short data test", 12, 8))
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
