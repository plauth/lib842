#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sw842.h>

static unsigned char workspace[1000000] = { 0 };

int init_module(void)
{
	unsigned char tmp[] = {
		0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33,
		0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37,
		0x38, 0x38, 0x39, 0x39, 0x40, 0x40, 0x41, 0x41,
		0x42, 0x42, 0x43, 0x43, 0x44, 0x44, 0x45, 0x45
	}; //"0011223344556677889900AABBCCDDEE";
	unsigned char out[64];
	unsigned char wtmp[32];
	int i = 0;
	unsigned cs = 64, ds = 32;
	char lbuf[300];

	printk(KERN_INFO "Module loaded.\n");

	memset(out, 0xff, 64);
	memset(wtmp, 0xff, 32);

	sw842_compress(tmp, 32, out, &cs, &workspace);
	for (i = 0; i < cs; i++)
		sprintf(&lbuf[3 * i], "%.2x:", out[i]);
	printk(KERN_INFO "Compressed: %s\n", lbuf);

	sw842_decompress(out, cs, wtmp, &ds);
	for (i = 0; i < 32; i++)
		sprintf(&lbuf[3 * i], "%.2x:", wtmp[i]);
	printk(KERN_INFO "Uncompressed: %s\n", lbuf);

	printk(KERN_INFO "Result: %s\n",
	       memcmp(tmp, wtmp, 32) == 0 ? "OK" : "KO");

	return 0;
}

void cleanup_module(void)
{
	printk(KERN_INFO "Module cleanup.\n");
}

MODULE_LICENSE("GPL");
MODULE_SOFTDEP("pre: 842");
