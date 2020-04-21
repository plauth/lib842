# nx-842 notes

This is a collection of notes related to the nx-842 kernel driver, which provides hardware accelerated compression for POWER hardware with a hardware compression accelerator.

## Enable debug output

To enable debug output from the hardware compressor driver available for POWER machines with an accelerated compressor, run the following as root:

echo 8 > /proc/sys/kernel/printk
echo 'file drivers/crypto/nx/nx-842-powernv.c +p' > /sys/kernel/debug/dynamic_debug/control
echo 'file drivers/crypto/nx/nx-842-pseries.c +p' > /sys/kernel/debug/dynamic_debug/control
echo 'file drivers/crypto/nx/nx-842.c +p' > /sys/kernel/debug/dynamic_debug/control
# Add more Linux kernel drivers files if you want

## Failures with empty and big blocks

This applies to the `nx_compress_powernv` kernel driver on kernel `4.15.0` at least.

The hardware kernel driver does not work with compressing an empty buffer (compress with `ilen=0`) or decompressing (a compressed input with no data) to an empty buffer (decompress with `olen=0`), which are operations that while a bit useless, could be done (and do work with the software kernel driver). Maybe those cases were not taken into consideration because the compressor can only be normally called internally by the kernel.

Similarly using big block sizes returns failures with some data where it could potentially work. For example compressing 256K of random data into a 512K or larger output buffer fails with ENOSPC. Internally I believe the kernel driver is supposed to support some bigger limit and split too big request into smaller ones, but it doesn't not seem to work after some point.

## Driver does not set `CRYPTO_ALG_KERN_DRIVER_ONLY` flag

The nx-842 kernel driver does not set the `CRYPTO_ALG_KERN_DRIVER_ONLY` flag which indicates that it is an accelerated kernel driver not available to userspace, which I believe should be set. This makes it a bit harder to detect when the accelerated hardware support or the software fallback is being used.

