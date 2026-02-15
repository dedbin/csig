#include <stddef.h>
#include <stdarg.h>

/* --- Простые функции --- */

int add(int a, int b) {
    return a + b;
}

void greet(const char *name) {
    (void)name;
}

/* --- Указатели и const --- */

int strlen_like(const char *s) {
    int len = 0;
    while (*s++) len++;
    return len;
}

void *memcpy_like(void *dest, const void *src, size_t n) {
    char *d = dest;
    const char *s = src;
    while (n--) {
        *d++ = *s++;
    }
    return dest;
}

/* --- Массивы --- */

int sum_array(int arr[], size_t n) {
    int sum = 0;
    for (size_t i = 0; i < n; i++)
        sum += arr[i];
    return sum;
}

/* --- typedef --- */

typedef unsigned long ulong_t;

ulong_t square_ul(ulong_t x) {
    return x * x;
}

/* --- Struct --- */

struct Point {
    double x;
    double y;
};

double distance_sq(struct Point *p) {
    return p->x * p->x + p->y * p->y;
}

/* --- Function pointer --- */

int apply(int (*func)(int, int), int a, int b) {
    return func(a, b);
}

/* --- Varargs --- */

int sum_variadic(int count, ...) {
    va_list args;
    va_start(args, count);

    int total = 0;
    for (int i = 0; i < count; i++) {
        total += va_arg(args, int);
    }

    va_end(args);
    return total;
}

/* --- Статическая --- */

static int internal_helper(double x) {
    return (int)x;
}
