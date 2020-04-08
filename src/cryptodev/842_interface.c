#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <crypto/cryptodev.h>

struct cryptodev_ctx {
	int cfd;
	struct session_op sess;
	uint16_t alignmask;
};

static int c842_ctx_init(struct cryptodev_ctx* ctx, int cfd)
{
#ifdef CIOCGSESSINFO
	struct session_info_op siop;
#endif

	memset(ctx, 0, sizeof(*ctx));
	ctx->cfd = cfd;

	ctx->sess.compr = CRYPTO_842;

	if (ioctl(ctx->cfd, CIOCGSESSION, &ctx->sess)) {
		fprintf(stderr, "ioctl(CIOCGSESSION) errno=%d\n", errno);
		return -errno;
	}

#ifdef CIOCGSESSINFO
	siop.ses = ctx->sess.ses;
	if (ioctl(ctx->cfd, CIOCGSESSINFO, &siop)) {
		fprintf(stderr, "ioctl(CIOCGSESSINFO) errno=%d\n", errno);
		return -errno;
	}
	#ifdef DEBUG
	printf("Got %s with driver %s\n",
			siop.compr_info.cra_name, siop.compr_info.cra_driver_name);
	if (!(siop.flags & SIOP_FLAG_KERNEL_DRIVER_ONLY)) {
		printf("Note: This is not an accelerated compressor\n");
	}
	#endif
	ctx->alignmask = siop.alignmask;
#endif
	return 0;
}

static int c842_ctx_deinit(struct cryptodev_ctx* ctx)
{
	if (ioctl(ctx->cfd, CIOCFSESSION, &ctx->sess.ses)) {
		fprintf(stderr, "ioctl(CIOCFSESSION) errno=%d\n", errno);
		return -errno;
	}
}

static int c842_compress(struct cryptodev_ctx* ctx, const void* input, size_t ilen, void* output, size_t *olen)
{
	struct crypt_op cryp;
	const void* p;
	
	/* check input and output alignment */
	if (ctx->alignmask) {
		p = (const void*)(((unsigned long)input + ctx->alignmask) & ~ctx->alignmask);
		if (input != p) {
			fprintf(stderr, "input is not aligned\n");
			return -EINVAL;
		}

		p = (const void*)(((unsigned long)output + ctx->alignmask) & ~ctx->alignmask);
		if (output != p) {
			fprintf(stderr, "output is not aligned\n");
			return -EINVAL;
		}
	}

	if (ilen > UINT32_MAX) {
		fprintf(stderr, "ilen too big\n");
		return -EINVAL;
	}
	if (*olen > UINT32_MAX) {
		fprintf(stderr, "olen too big\n");
		return -EINVAL;
	}

	memset(&cryp, 0, sizeof(cryp));

	/* Encrypt data.in to data.encrypted */
	cryp.ses = ctx->sess.ses;
	cryp.len = (__u32)ilen;
	cryp.dlen = (__u32)*olen;
	cryp.src = (__u8*)input;
	cryp.dst = (__u8*)output;
	cryp.op = COP_ENCRYPT;
	if (ioctl(ctx->cfd, CIOCCRYPT, &cryp)) {
		fprintf(stderr, "ioctl(CIOCCRYPT) errno=%d\n", errno);
		return -errno;
	}

	*olen = cryp.dlen;

	return 0;
}

static int c842_decompress(struct cryptodev_ctx* ctx, const void* input, size_t ilen, void* output, size_t *olen)
{
	struct crypt_op cryp;
	const void* p;
	
	/* check input and output alignment */
	if (ctx->alignmask) {
		p = (const void*)(((unsigned long)input + ctx->alignmask) & ~ctx->alignmask);
		if (input != p) {
			fprintf(stderr, "input is not aligned\n");
			return -EINVAL;
		}

		p = (const void*)(((unsigned long)output + ctx->alignmask) & ~ctx->alignmask);
		if (output != p) {
			fprintf(stderr, "output is not aligned\n");
			return -EINVAL;
		}
	}

	if (ilen > UINT32_MAX) {
		fprintf(stderr, "ilen too big\n");
		return -EINVAL;
	}
	if (*olen > UINT32_MAX) {
		fprintf(stderr, "olen too big\n");
		return -EINVAL;
	}

	memset(&cryp, 0, sizeof(cryp));

	/* Encrypt data.in to data.encrypted */
	cryp.ses = ctx->sess.ses;
	cryp.len = (__u32)ilen;
	cryp.dlen = (__u32)*olen;
	cryp.src = (__u8*)input;
	cryp.dst = (__u8*)output;
	cryp.op = COP_DECRYPT;
	if (ioctl(ctx->cfd, CIOCCRYPT, &cryp)) {
		fprintf(stderr, "ioctl(CIOCCRYPT) errno=%i\n", errno);
		return -errno;
	}

	*olen = cryp.dlen;

	return 0;
}

int hw842_compress(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen) {
	int cfd;
	int err = 0, cleanup_err = 0;
	struct cryptodev_ctx ctx;

	/* Open the crypto device */
	cfd = open("/dev/crypto", O_RDWR, 0);
	if (cfd < 0) {
		fprintf(stderr, "open(/dev/crypto)\n");
		return -errno;
	}

	/* Set close-on-exec (not really needed here) */
	if (fcntl(cfd, F_SETFD, 1) == -1) {
		fprintf(stderr, "fcntl(F_SETFD)\n");
		err = -errno;
		goto cleanup_file;
	}

	err = c842_ctx_init(&ctx, cfd);
	if(err)
		goto cleanup_file;

	err = c842_compress(&ctx, in, ilen, out, olen);

	cleanup_err = c842_ctx_deinit(&ctx);
	if (err == 0)
		err = cleanup_err;

cleanup_file:
	/* Close the original descriptor */
	if (close(cfd)) {
		fprintf(stderr, "close(cfd)\n");
		if (err == 0)
			err = -errno;
	}

	return err;
}

int hw842_decompress(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen) {
	int cfd;
	int err = 0, cleanup_err = 0;
	struct cryptodev_ctx ctx;

	/* Open the crypto device */
	cfd = open("/dev/crypto", O_RDWR, 0);
	if (cfd < 0) {
		fprintf(stderr, "open(/dev/crypto)\n");
		return -errno;
	}

	/* Set close-on-exec (not really needed here) */
	if (fcntl(cfd, F_SETFD, 1) == -1) {
		fprintf(stderr, "fcntl(F_SETFD)\n");
		err = -errno;
		goto cleanup_file;
	}

	err = c842_ctx_init(&ctx, cfd);
	if(err)
		goto cleanup_file;

	err = c842_decompress(&ctx, in, ilen, out, olen);

	cleanup_err = c842_ctx_deinit(&ctx);
	if (err == 0)
		err = cleanup_err;

cleanup_file:
	/* Close the original descriptor */
	if (close(cfd)) {
		fprintf(stderr, "close(cfd)\n");
		if (err == 0)
			err = -errno;
	}

	return err;
}