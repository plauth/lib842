#define BITS_PER_LONG_LONG 64
#define GENMASK_ULL(h, l) \
	(((~0ULL) << (l)) & (~0ULL >> (BITS_PER_LONG_LONG - 1 - (h))))
		
#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define round_up(x, y) ((((x)-1) | __round_mask(x, y))+1)
#define round_down(x, y) ((x) & ~__round_mask(x, y))

/*
round_up(0, 8) = 0
round_up(1, 8) = 8
round_up(2, 8) = 8
round_up(3, 8) = 8
round_up(4, 8) = 8
round_up(5, 8) = 8
round_up(6, 8) = 8
round_up(7, 8) = 8
round_up(8, 8) = 8
round_up(9, 8) = 16
round_up(10, 8) = 16
round_up(11, 8) = 16
round_up(12, 8) = 16
round_up(13, 8) = 16
round_up(14, 8) = 16
round_up(15, 8) = 16
round_up(16, 8) = 16
round_up(17, 8) = 24
round_up(18, 8) = 24
round_up(19, 8) = 24
round_up(20, 8) = 24
round_up(21, 8) = 24
round_up(22, 8) = 24
round_up(23, 8) = 24
round_up(24, 8) = 24
round_up(25, 8) = 32
round_up(26, 8) = 32
round_up(27, 8) = 32
round_up(28, 8) = 32
round_up(29, 8) = 32
round_up(30, 8) = 32
round_up(31, 8) = 32 


round_down(0, 8) = 0
round_down(1, 8) = 0
round_down(2, 8) = 0
round_down(3, 8) = 0
round_down(4, 8) = 0
round_down(5, 8) = 0
round_down(6, 8) = 0
round_down(7, 8) = 0
round_down(8, 8) = 8
round_down(9, 8) = 8
round_down(10, 8) = 8
round_down(11, 8) = 8
round_down(12, 8) = 8
round_down(13, 8) = 8
round_down(14, 8) = 8
round_down(15, 8) = 8
round_down(16, 8) = 16
round_down(17, 8) = 16
round_down(18, 8) = 16
round_down(19, 8) = 16
round_down(20, 8) = 16
round_down(21, 8) = 16
round_down(22, 8) = 16
round_down(23, 8) = 16
round_down(24, 8) = 24
round_down(25, 8) = 24
round_down(26, 8) = 24
round_down(27, 8) = 24
round_down(28, 8) = 24
round_down(29, 8) = 24
round_down(30, 8) = 24
round_down(31, 8) = 24
*/
	
		
