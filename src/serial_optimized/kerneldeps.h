#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define round_down(x, y) ((x) & ~__round_mask(x, y))