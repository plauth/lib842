#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#ifndef __STDC_NO_THREADS__
#include <threads.h>
#else
// Even though all systems we are currently targetting support threads,
// implementations such as glibc didn't support for C11 threads until recently
// But since C11 threads are almost a clone of pthreads, we can easily fall
// back to those if C11 threads are not supported but pthreads is
#include <pthread.h>
#define ONCE_FLAG_INIT PTHREAD_ONCE_INIT
#define once_flag pthread_once_t
#define call_once pthread_once
#define tss_t pthread_key_t
#define tss_create pthread_key_create
#define thrd_success 0
#define tss_set pthread_setspecific
#define tss_get pthread_getspecific
#endif
#include <sys/ioctl.h>
#include <crypto/cryptodev.h>
#include <lib842/hw.h>

struct cryptodev_ctx {
	int cfd;
	struct session_op sess;
	uint16_t alignmask;
};

static int c842_ctx_init(struct cryptodev_ctx *ctx)
{
	int err;

	/* Open the crypto device */
	ctx->cfd = open("/dev/crypto", O_RDWR, 0);
	if (ctx->cfd < 0) {
		err = -errno;
		fprintf(stderr, "open(/dev/crypto) failed (%d): %s\n",
			errno, strerror(errno));
		goto return_error;
	}

	/* Set close-on-exec (not really needed here) */
	if (fcntl(ctx->cfd, F_SETFD, 1) == -1) {
		err = -errno;
		fprintf(stderr, "fcntl(F_SETFD) failed (%d): %s\n",
			errno, strerror(errno));
		goto cleanup_file;
	}

	memset(&ctx->sess, 0, sizeof(ctx->sess));
	ctx->sess.compr = CRYPTO_842;

	if (ioctl(ctx->cfd, CIOCGSESSION, &ctx->sess)) {
		err = -errno;
		fprintf(stderr, "ioctl(CIOCGSESSION) failed (%d): %s\n",
			errno, strerror(errno));
		goto cleanup_file;
	}

#ifdef CIOCGSESSINFO
	struct session_info_op siop = { 0 };
	siop.ses = ctx->sess.ses;
	if (ioctl(ctx->cfd, CIOCGSESSINFO, &siop)) {
		err = -errno;
		fprintf(stderr, "ioctl(CIOCGSESSINFO) failed (%d): %s\n",
			errno, strerror(errno));
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
		fprintf(stderr, "ioctl(CIOCFSESSION) failed (%d): %s\n",
			errno, strerror(errno));
	}

	/* Close the original descriptor */
	if (close(ctx->cfd)) {
		if (err == 0)
			err = -errno;
		fprintf(stderr, "close(cfd) failed (%d): %s\n",
			errno, strerror(errno));
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
		fprintf(stderr, "ioctl(CIOCCRYPT) failed (%d): %s\n",
			errno, strerror(errno));
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
		int err = -errno;
		fprintf(stderr, "ioctl(CIOCCRYPT) failed (%d): %s\n",
			errno, strerror(errno));
		return err;
	}

	*olen = cryp.dlen;

	return 0;
}

// From the outside, we present an easy-to-use interface with only two
// functions (compress and decompress). For performance, we create just a
// cryptodev context per thread, which is stored in a thread-local variable
static tss_t thread_ctx_key;
static int thread_ctx_key_err;
static once_flag thread_ctx_key_once = ONCE_FLAG_INIT;

static void destroy_thread_ctx(void *ctx)
{
	// We can't do anything with errors here, so ignore them
	(void)c842_ctx_deinit((struct cryptodev_ctx *)ctx);
	free(ctx);
}

static void create_thread_ctx_key()
{
	if (tss_create(&thread_ctx_key, destroy_thread_ctx) != thrd_success)
		thread_ctx_key_err = -ENOMEM;

	thread_ctx_key_err = 0;
}

static int get_thread_cryptodev_ctx(struct cryptodev_ctx **rctx)
{
	call_once(&thread_ctx_key_once, create_thread_ctx_key);
	if (thread_ctx_key_err)
		return thread_ctx_key_err;

	struct cryptodev_ctx *ctx = tss_get(thread_ctx_key);
	if (ctx != NULL) { // Context already created for this thread
		*rctx = ctx;
		return 0;
	}

	ctx = malloc(sizeof(struct cryptodev_ctx));
	if (ctx == NULL)
		return -ENOMEM;

	int err = c842_ctx_init(ctx);
	if (err) {
		free(ctx);
		return err;
	}

	if (tss_set(thread_ctx_key, ctx) != thrd_success) {
		c842_ctx_deinit(ctx);
		free(ctx);
		return -ENOMEM;
	}

	*rctx = ctx;
	return 0;
}

int hw842_compress(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen)
{
	struct cryptodev_ctx *thread_ctx;
	int err = get_thread_cryptodev_ctx(&thread_ctx);
	if (err)
		return err;

	return c842_compress(thread_ctx, in, ilen, out, olen);
}

int hw842_decompress(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen)
{
	struct cryptodev_ctx *thread_ctx;
	int err = get_thread_cryptodev_ctx(&thread_ctx);
	if (err)
		return err;

	return c842_decompress(thread_ctx, in, ilen, out, olen);
}
