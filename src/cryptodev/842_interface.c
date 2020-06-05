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
#include <unistd.h>

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

static int c842_compress_chunked(struct cryptodev_ctx *ctx, __u16 op, size_t numchunks,
				 const uint8_t *in, size_t isize, const size_t *ilens,
				 uint8_t *out, size_t osize, size_t *olens)
{
	struct crypt_op cryp = { 0 };
	__u32 ilens32[CRYPTODEV_COMP_MAX_CHUNKS],
	      olens32[CRYPTODEV_COMP_MAX_CHUNKS];

	/* check input and output alignment */
	if (ctx->alignmask && !is_pointer_aligned(in, ctx->alignmask)) {
		fprintf(stderr, "in is not aligned\n");
		return -EINVAL;
	}
	if (ctx->alignmask && !is_pointer_aligned(out, ctx->alignmask)) {
		fprintf(stderr, "out is not aligned\n");
		return -EINVAL;
	}

	if (isize > UINT32_MAX) {
		fprintf(stderr, "isize too big\n");
		return -EINVAL;
	}
	if (osize > UINT32_MAX) {
		fprintf(stderr, "osize too big\n");
		return -EINVAL;
	}

	if (numchunks > CRYPTODEV_COMP_MAX_CHUNKS) {
		fprintf(stderr, "numchunks too big\n");
		return -EINVAL;
	}

	for (size_t i = 0; i < numchunks; i++) {
		if (ilens[i] > UINT32_MAX) {
			fprintf(stderr, "ilens[%zu] too big\n", i);
			return -EINVAL;
		}
		if (olens[i] > UINT32_MAX) {
			fprintf(stderr, "olens[%zu] too big\n", i);
			return -EINVAL;
		}

		ilens32[i] = (__u32)ilens[i];
		olens32[i] = (__u32)olens[i];
	}

	/* Encrypt data.in to data.encrypted */
	cryp.ses = ctx->sess.ses;
	cryp.op = op;
	cryp.len = (__u32)isize;
	cryp.dlen = (__u32)osize;
	cryp.src = (__u8 *)in;
	cryp.dst = out;
	cryp.numchunks = numchunks;
	cryp.chunklens = ilens32;
	cryp.chunkdlens = olens32;
	if (ioctl(ctx->cfd, CIOCCRYPT, &cryp)) {
		int err = -errno;
		fprintf(stderr, "ioctl(CIOCCRYPT) failed (%d): %s\n",
			errno, strerror(errno));
		return err;
	}

	for (size_t i = 0; i < numchunks; i++) {
		olens[i] = olens32[i];
	}

	return 0;
}

static int c842_compress(struct cryptodev_ctx *ctx, __u16 op,
			 const uint8_t *in, size_t ilen,
			 uint8_t *out, size_t *olen)
{
	return c842_compress_chunked(ctx, op, 1, in, ilen, &ilen, out, *olen, olen);
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

static size_t hw842_get_required_alignment() {
	struct cryptodev_ctx *thread_ctx;
	int err = get_thread_cryptodev_ctx(&thread_ctx);
	if (err) {
		// Maximum alignment (note alignmask is a uint16_t)
		// It will likely fail again later anyway
		return 0x10000;
	}

	return thread_ctx->alignmask + 1;
}

int hw842_available() {
	// TODO: A self-test trying to decompress a known bitstream would be better here
	return access("/dev/crypto", F_OK) == 0;
}

int hw842_compress(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen)
{
	struct cryptodev_ctx *thread_ctx;
	int err = get_thread_cryptodev_ctx(&thread_ctx);
	if (err)
		return err;

	return c842_compress(thread_ctx, COP_ENCRYPT, in, ilen, out, olen);
}

int hw842_decompress(const uint8_t *in, size_t ilen, uint8_t *out, size_t *olen)
{
	struct cryptodev_ctx *thread_ctx;
	int err = get_thread_cryptodev_ctx(&thread_ctx);
	if (err)
		return err;

	return c842_compress(thread_ctx, COP_DECRYPT, in, ilen, out, olen);
}

int hw842_compress_chunked(size_t numchunks,
			   const uint8_t *in, size_t isize, const size_t *ilens,
			   uint8_t *out, size_t osize, size_t *olens)
{
	struct cryptodev_ctx *thread_ctx;
	int err = get_thread_cryptodev_ctx(&thread_ctx);
	if (err)
		return err;

	return c842_compress_chunked(thread_ctx, COP_ENCRYPT, numchunks,
				     in, isize, ilens,
				     out, osize, olens);
}

int hw842_decompress_chunked(size_t numchunks,
			     const uint8_t *in, size_t isize, const size_t *ilens,
			     uint8_t *out, size_t osize, size_t *olens)
{
	struct cryptodev_ctx *thread_ctx;
	int err = get_thread_cryptodev_ctx(&thread_ctx);
	if (err)
		return err;

	return c842_compress_chunked(thread_ctx, COP_DECRYPT, numchunks,
				     in, isize, ilens,
				     out, osize, olens);
}

const struct lib842_implementation *get_hw842_implementation() {
	static struct lib842_implementation hw842_implementation = {
		.compress = hw842_compress,
		.decompress = hw842_decompress,
		.compress_chunked = hw842_compress_chunked,
		.decompress_chunked = hw842_decompress_chunked,
		.required_alignment = 1,
		.preferred_alignment = 0
	};
	if (hw842_implementation.required_alignment == 0)
		hw842_implementation.required_alignment = hw842_get_required_alignment();
	// The cryptodev implementation can work better (do zero copy)
	// if the buffers are page-aligned
	if (hw842_implementation.preferred_alignment == 0)
		hw842_implementation.preferred_alignment = sysconf(_SC_PAGESIZE);

	return &hw842_implementation;
}
