/* 
 * Computes the floored log2 of a given value at compile-time.
 * The result is rounded down to the next base-2 boundary.
 *
 * Examples:
 *
 * #define STATIC_LOG2_ARG 89
 * #include static_log2.h
 * char myarray[STATIC_LOG2_VALUE]; // sizeof(myarray) == 6
 *
 * #define STATIC_LOG2_ARG 8
 * #include static_log2.h
 * char myarray[STATIC_LOG2_VALUE]; // sizeof(myarray) == 3
 *
 * #define STATIC_LOG2_ARG 1234 
 * #include static_log2.h
 * char myarray[STATIC_LOG2_VALUE]; // sizeof(myarray) == 10
 *
 */

#ifndef STATIC_LOG2_ARG
#error "Please define STATIC_LOG2_ARG"
#endif

#undef STATIC_LOG2_VALUE

#if STATIC_LOG2_ARG & (1 << 31)
#define STATIC_LOG2_VALUE 31

#elif STATIC_LOG2_ARG & (1 << 30)
#define STATIC_LOG2_VALUE 30

#elif STATIC_LOG2_ARG & (1 << 29)
#define STATIC_LOG2_VALUE 29 

#elif STATIC_LOG2_ARG & (1 << 28)
#define STATIC_LOG2_VALUE 28 

#elif STATIC_LOG2_ARG & (1 << 27)
#define STATIC_LOG2_VALUE 27

#elif STATIC_LOG2_ARG & (1 << 26)
#define STATIC_LOG2_VALUE 26

#elif STATIC_LOG2_ARG & (1 << 25)
#define STATIC_LOG2_VALUE 25

#elif STATIC_LOG2_ARG & (1 << 24)
#define STATIC_LOG2_VALUE 24

#elif STATIC_LOG2_ARG & (1 << 23)
#define STATIC_LOG2_VALUE 23

#elif STATIC_LOG2_ARG & (1 << 22)
#define STATIC_LOG2_VALUE 22

#elif STATIC_LOG2_ARG & (1 << 21)
#define STATIC_LOG2_VALUE 21

#elif STATIC_LOG2_ARG & (1 << 20)
#define STATIC_LOG2_VALUE 20

#elif STATIC_LOG2_ARG & (1 << 19)
#define STATIC_LOG2_VALUE 19

#elif STATIC_LOG2_ARG & (1 << 18)
#define STATIC_LOG2_VALUE 18

#elif STATIC_LOG2_ARG & (1 << 17)
#define STATIC_LOG2_VALUE 17

#elif STATIC_LOG2_ARG & (1 << 16)
#define STATIC_LOG2_VALUE 16

#elif STATIC_LOG2_ARG & (1 << 15)
#define STATIC_LOG2_VALUE 15

#elif STATIC_LOG2_ARG & (1 << 14)
#define STATIC_LOG2_VALUE 14

#elif STATIC_LOG2_ARG & (1 << 13)
#define STATIC_LOG2_VALUE 13

#elif STATIC_LOG2_ARG & (1 << 12)
#define STATIC_LOG2_VALUE 12

#elif STATIC_LOG2_ARG & (1 << 11)
#define STATIC_LOG2_VALUE 11

#elif STATIC_LOG2_ARG & (1 << 10)
#define STATIC_LOG2_VALUE 10

#elif STATIC_LOG2_ARG & (1 << 9)
#define STATIC_LOG2_VALUE 9

#elif STATIC_LOG2_ARG & (1 << 8)
#define STATIC_LOG2_VALUE 8

#elif STATIC_LOG2_ARG & (1 << 7)
#define STATIC_LOG2_VALUE 7

#elif STATIC_LOG2_ARG & (1 << 6)
#define STATIC_LOG2_VALUE 6

#elif STATIC_LOG2_ARG & (1 << 5)
#define STATIC_LOG2_VALUE 5

#elif STATIC_LOG2_ARG & (1 << 4)
#define STATIC_LOG2_VALUE 4

#elif STATIC_LOG2_ARG & (1 << 3)
#define STATIC_LOG2_VALUE 3

#elif STATIC_LOG2_ARG & (1 << 2)
#define STATIC_LOG2_VALUE 2

#elif STATIC_LOG2_ARG & (1 << 1)
#define STATIC_LOG2_VALUE 0

#else
#define STATIC_LOG2_VALUE -1 
#endif

/* allow multiple inclusion of this header. */
#undef STATIC_LOG2_ARG
