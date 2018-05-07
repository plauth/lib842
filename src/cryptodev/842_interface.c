#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <crypto/cryptodev.h>

struct cryptodev_ctx {
	int cfd;
	struct session_op sess;
	uint16_t alignmask;
};

int c842_ctx_init(struct cryptodev_ctx* ctx, int cfd)
{
#ifdef CIOCGSESSINFO
	struct session_info_op siop;
#endif

	memset(ctx, 0, sizeof(*ctx));
	ctx->cfd = cfd;

	ctx->sess.compr = CRYPTO_842;

	if (ioctl(ctx->cfd, CIOCGSESSION, &ctx->sess)) {
		fprintf(stderr, "ioctl(CIOCGSESSION)\n");
		return -1;
	}

#ifdef CIOCGSESSINFO
	siop.ses = ctx->sess.ses;
	if (ioctl(ctx->cfd, CIOCGSESSINFO, &siop)) {
		fprintf(stderr, "ioctl(CIOCGSESSINFO)\n");
		return -1;
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

void c842_ctx_deinit(struct cryptodev_ctx* ctx) 
{
	if (ioctl(ctx->cfd, CIOCFSESSION, &ctx->sess.ses)) {
		fprintf(stderr, "ioctl(CIOCFSESSION)\n");
		exit(-1);
	}
}

int c842_compress(struct cryptodev_ctx* ctx, const void* input, unsigned int ilen, void* output, unsigned int *olen)
{
	struct crypt_op cryp;
	void* p;
	
	/* check input and output alignment */
	if (ctx->alignmask) {
		p = (void*)(((unsigned long)input + ctx->alignmask) & ~ctx->alignmask);
		if (input != p) {
			fprintf(stderr, "input is not aligned\n");
			return -1;
		}

		p = (void*)(((unsigned long)output + ctx->alignmask) & ~ctx->alignmask);
		if (output != p) {
			fprintf(stderr, "output is not aligned\n");
			return -1;
		}
	}

	memset(&cryp, 0, sizeof(cryp));

	/* Encrypt data.in to data.encrypted */
	cryp.ses = ctx->sess.ses;
	cryp.len = ilen;
	cryp.dlen = *olen;
	cryp.src = (__u8*)input;
	cryp.dst = (__u8*)output;
	cryp.op = COP_ENCRYPT;
	if (ioctl(ctx->cfd, CIOCCRYPT, &cryp)) {
		fprintf(stderr, "ioctl(CIOCCRYPT)\n");
		return -1;
	}

	*olen = cryp.dlen;

	return 0;
}

int c842_decompress(struct cryptodev_ctx* ctx, const void* input, unsigned int ilen, void* output, unsigned int *olen)
{
	struct crypt_op cryp;
	void* p;
	
	/* check input and output alignment */
	if (ctx->alignmask) {
		p = (void*)(((unsigned long)input + ctx->alignmask) & ~ctx->alignmask);
		if (input != p) {
			fprintf(stderr, "input is not aligned\n");
			return -1;
		}

		p = (void*)(((unsigned long)output + ctx->alignmask) & ~ctx->alignmask);
		if (output != p) {
			fprintf(stderr, "output is not aligned\n");
			return -1;
		}
	}

	memset(&cryp, 0, sizeof(cryp));

	/* Encrypt data.in to data.encrypted */
	cryp.ses = ctx->sess.ses;
	cryp.len = ilen;
	cryp.dlen = *olen;
	cryp.src = (__u8*)input;
	cryp.dst = (__u8*)output;
	cryp.op = COP_DECRYPT;
	if (ioctl(ctx->cfd, CIOCCRYPT, &cryp)) {
		fprintf(stderr, "ioctl(CIOCCRYPT)\n");
		return -1;
	}

	*olen = cryp.dlen;

	return 0;
}

int hw842_compress(const uint8_t *in, unsigned int ilen, uint8_t *out, unsigned int *olen) {
	int cfd = -1;
	int err = 0;
	struct cryptodev_ctx ctx;

	/* Open the crypto device */
	cfd = open("/dev/crypto", O_RDWR, 0);
	if (cfd < 0) {
		fprintf(stderr, "open(/dev/crypto)\n");
		exit(-1);
	}

	/* Set close-on-exec (not really needed here) */
	if (fcntl(cfd, F_SETFD, 1) == -1) {
		fprintf(stderr, "fcntl(F_SETFD)\n");
		exit(-1);
	}

	err = c842_ctx_init(&ctx, cfd);
	if(err)
		exit(-1);

	err = c842_compress(&ctx, in, ilen, out, olen);
	if(err)
		exit(-1);

	c842_ctx_deinit(&ctx);

	/* Close the original descriptor */
	if (close(cfd)) {
		fprintf(stderr, "close(cfd)\n");
		return 1;
	}

	return 0;
}

int hw842_decompress(const uint8_t *in, unsigned int ilen, uint8_t *out, unsigned int *olen) {
	int cfd = -1;
	int err = 0;
	struct cryptodev_ctx ctx;

	/* Open the crypto device */
	cfd = open("/dev/crypto", O_RDWR, 0);
	if (cfd < 0) {
		fprintf(stderr, "open(/dev/crypto)\n");
		exit(-1);
	}

	/* Set close-on-exec (not really needed here) */
	if (fcntl(cfd, F_SETFD, 1) == -1) {
		fprintf(stderr, "fcntl(F_SETFD)\n");
		exit(-1);
	}

	err = c842_ctx_init(&ctx, cfd);
	if(err)
		exit(-1);

	err = c842_decompress(&ctx, in, ilen, out, olen);
	if(err)
		exit(-1);

	c842_ctx_deinit(&ctx);

	/* Close the original descriptor */
	if (close(cfd)) {
		fprintf(stderr, "close(cfd)\n");
		return 1;
	}

	return 0;
}