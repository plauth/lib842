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
#include <pthread.h>
#define mtx_t pthread_mutex_t
#define mtx_lock pthread_mutex_lock
#define mtx_unlock pthread_mutex_unlock
#define thread_local __thread
#endif
#include <stdbool.h>
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

// Unfortunately, there doesn't seem to be any really nice way to
// release the thread local cryptodev context at thread exit without
// the help of the caller. As an alternative, store all the contexts
// in a global list which will get released at program exit (atexit)
static mtx_t release_contexts_mutex;
static struct cryptodev_ctx **release_contexts_list = NULL;
static size_t release_contexts_count = 0;
static size_t release_contexts_capacity = 0;

static void atexit_release_cryptodev_contexts()
{
	for (size_t i = 0; i < release_contexts_count; i++)
		c842_ctx_deinit(release_contexts_list[i]);
	free(release_contexts_list);
}

static int register_cryptodev_context(struct cryptodev_ctx *ctx) {
	mtx_lock(&release_contexts_mutex);

	if (release_contexts_capacity == release_contexts_count) {
		size_t new_capacity = release_contexts_capacity != 0
			? release_contexts_capacity * 2
			: 4;
		struct cryptodev_ctx **new_list = realloc(release_contexts_list,
			new_capacity * sizeof(struct cryptodev_ctx *));
		if (new_list == NULL) {
			mtx_unlock(&release_contexts_mutex);
			return -ENOMEM;
		}

		release_contexts_list = new_list;
		release_contexts_capacity = new_capacity;
	}

	if (release_contexts_count == 0) {
		if (atexit(atexit_release_cryptodev_contexts) != 0) {
			mtx_unlock(&release_contexts_mutex);
			return -ENOMEM;
		}
	}

	release_contexts_list[release_contexts_count++] = ctx;
	mtx_unlock(&release_contexts_mutex);

	return 0;
}

static thread_local bool have_thread_ctx = false;
static thread_local struct cryptodev_ctx thread_ctx;

static int ensure_thread_ctx_exists()
{
	if (have_thread_ctx)
		return 0;

	int err = c842_ctx_init(&thread_ctx);
	if (err)
		return err;

	err = register_cryptodev_context(&thread_ctx);
	if (err) {
		c842_ctx_deinit(&thread_ctx);
		return err;
	}

	have_thread_ctx = true;
	return 0;
}

int hw842_compress(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen)
{
	int err = ensure_thread_ctx_exists();
	if (err)
		return err;

	return c842_compress(&thread_ctx, in, ilen, out, olen);
}

int hw842_decompress(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen)
{
	int err = ensure_thread_ctx_exists();
	if (err)
		return err;

	return c842_decompress(&thread_ctx, in, ilen, out, olen);
}