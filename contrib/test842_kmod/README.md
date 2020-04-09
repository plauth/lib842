This is a small Linux kernel module that can be used to call sw842_(de)compress, in order to verify the reference implementation.

Build the module with the given Makefile (you will need to have the Kernel headers installed) and then load the module with 'insmod test842.ko'. The output of the test will be printed to the kernel log (dmesg).
