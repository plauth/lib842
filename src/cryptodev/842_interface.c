#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <crypto/cryptodev.h>
#include "../../include/hw842.h"

struct cryptodev_ctx {
	int cfd;
	struct session_op sess;
	uint16_t alignmask;
};

static int c842_ctx_init(struct cryptodev_ctx *ctx)
{
	int err = 0;

	/* Open the crypto device */
	ctx->cfd = open("/dev/crypto", O_RDWR, 0);
	if (ctx->cfd < 0) {
		err = -errno;
		fprintf(stderr, "open(/dev/crypto) errno=%d\n", errno);
		goto return_error;
	}

	/* Set close-on-exec (not really needed here) */
	if (fcntl(ctx->cfd, F_SETFD, 1) == -1) {
		err = -errno;
		fprintf(stderr, "fcntl(F_SETFD) errno=%d\n", errno);
		goto cleanup_file;
	}

	memset(&ctx->sess, 0, sizeof(ctx->sess));
	ctx->sess.compr = CRYPTO_842;

	if (ioctl(ctx->cfd, CIOCGSESSION, &ctx->sess)) {
		err = -errno;
		fprintf(stderr, "ioctl(CIOCGSESSION) errno=%d\n", errno);
		goto cleanup_file;
	}

#ifdef CIOCGSESSINFO
	struct session_info_op siop = { 0 };
	siop.ses = ctx->sess.ses;
	if (ioctl(ctx->cfd, CIOCGSESSINFO, &siop)) {
		err = -errno;
		fprintf(stderr, "ioctl(CIOCGSESSINFO) errno=%d\n", errno);
		goto cleanup_file_and_session;
	}
#ifdef DEBUG
	printf("Got %s with driver %s\n", siop.compr_info.cra_name,
	       siop.compr_info.cra_driver_name);
	if (!(siop.flags & SIOP_FLAG_KERNEL_DRIVER_ONLY)) {
		printf("Note: This is not an accelerated compressor\n");
	}
#endif
	ctx->alignmask = siop.alignmask;
#endif

	return 0;

cleanup_file_and_session:
	ioctl(ctx->cfd, CIOCFSESSION, &ctx->sess.ses);
cleanup_file:
	close(ctx->cfd);
return_error:
	return err;
}

static int c842_ctx_deinit(struct cryptodev_ctx *ctx)
{
	int err = 0;

	if (ioctl(ctx->cfd, CIOCFSESSION, &ctx->sess.ses)) {
		err = -errno;
		fprintf(stderr, "ioctl(CIOCFSESSION) errno=%d\n", errno);
	}

	/* Close the original descriptor */
	if (close(ctx->cfd)) {
		if (err == 0)
			err = -errno;
		fprintf(stderr, "close(cfd) errno=%d\n", errno);
	}

	return err;
}

static int is_pointer_aligned(const void *ptr, uint16_t alignmask)
{
	const void *aligned_ptr =
		(const void *)(((unsigned long)ptr + alignmask) & ~alignmask);
	return ptr == aligned_ptr;
}

static int c842_compress(struct cryptodev_ctx *ctx, const void *input,
			 size_t ilen, void *output, size_t *olen)
{
	struct crypt_op cryp = { 0 };

	/* check input and output alignment */
	if (ctx->alignmask && !is_pointer_aligned(input, ctx->alignmask)) {
		fprintf(stderr, "input is not aligned\n");
		return -EINVAL;
	}
	if (ctx->alignmask && !is_pointer_aligned(output, ctx->alignmask)) {
		fprintf(stderr, "output is not aligned\n");
		return -EINVAL;
	}

	if (ilen > UINT32_MAX) {
		fprintf(stderr, "ilen too big\n");
		return -EINVAL;
	}
	if (*olen > UINT32_MAX) {
		fprintf(stderr, "olen too big\n");
		return -EINVAL;
	}

	/* Encrypt data.in to data.encrypted */
	cryp.ses = ctx->sess.ses;
	cryp.len = (__u32)ilen;
	cryp.dlen = (__u32)*olen;
	cryp.src = (__u8 *)input;
	cryp.dst = (__u8 *)output;
	cryp.op = COP_ENCRYPT;
	if (ioctl(ctx->cfd, CIOCCRYPT, &cryp)) {
		fprintf(stderr, "ioctl(CIOCCRYPT) errno=%d\n", errno);
		return -errno;
	}

	*olen = cryp.dlen;

	return 0;
}

static int c842_decompress(struct cryptodev_ctx *ctx, const void *input,
			   size_t ilen, void *output, size_t *olen)
{
	struct crypt_op cryp = { 0 };

	/* check input and output alignment */
	if (ctx->alignmask && !is_pointer_aligned(input, ctx->alignmask)) {
		fprintf(stderr, "input is not aligned\n");
		return -EINVAL;
	}
	if (ctx->alignmask && !is_pointer_aligned(output, ctx->alignmask)) {
		fprintf(stderr, "output is not aligned\n");
		return -EINVAL;
	}

	if (ilen > UINT32_MAX) {
		fprintf(stderr, "ilen too big\n");
		return -EINVAL;
	}
	if (*olen > UINT32_MAX) {
		fprintf(stderr, "olen too big\n");
		return -EINVAL;
	}

	/* Encrypt data.in to data.encrypted */
	cryp.ses = ctx->sess.ses;
	cryp.len = (__u32)ilen;
	cryp.dlen = (__u32)*olen;
	cryp.src = (__u8 *)input;
	cryp.dst = (__u8 *)output;
	cryp.op = COP_DECRYPT;
	if (ioctl(ctx->cfd, CIOCCRYPT, &cryp)) {
		fprintf(stderr, "ioctl(CIOCCRYPT) errno=%i\n", errno);
		return -errno;
	}

	*olen = cryp.dlen;

	return 0;
}

int hw842_compress(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen)
{
	int err = 0, cleanup_err = 0;
	struct cryptodev_ctx ctx;

	err = c842_ctx_init(&ctx);
	if (err)
		return err;

	err = c842_compress(&ctx, in, ilen, out, olen);

	cleanup_err = c842_ctx_deinit(&ctx);
	if (err == 0)
		err = cleanup_err;

	return err;
}

int hw842_decompress(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen)
{
	int err = 0, cleanup_err = 0;
	struct cryptodev_ctx ctx;

	err = c842_ctx_init(&ctx);
	if (err)
		return err;

	err = c842_decompress(&ctx, in, ilen, out, olen);

	cleanup_err = c842_ctx_deinit(&ctx);
	if (err == 0)
		err = cleanup_err;

	return err;
}