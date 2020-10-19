#pragma once

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Placeholder types
typedef signed long int int64_t;
typedef unsigned long int uint64_t;
typedef unsigned long int size_t;
typedef unsigned long int uintptr_t;

/*
 * Compares two strings up to and including either the null terminator or the n-th character (max n characters total).
 * Returns true if equal.
 * As a replaceholder for "strncmp(...) == 0".
 */
__host__ __device__ bool local_strneq(const char *str1, const char *str2, size_t n);

/*
 * Like strtoul, but always base 10 numbers.
 */
__host__ __device__ unsigned long int local_strtoul10(const char *str, char **endptr);

/*
 * See strcspn.
 */
__host__ __device__ size_t local_strcspn(const char *str1, const char *str2);

/*
 * See strlen.
 */
__host__ __device__ size_t local_strlen(const char *str);

/*
 * See strnlen.
 */
__host__ __device__ size_t local_strnlen(const char *str, size_t n);

/*
 * See strncpy.
 */
__host__ __device__ char *local_strncpy(char *dst, const char *src, size_t n);

/*
 * See __stpncpy.
 */
__host__ __device__ char *local_stpncpy(char *dst, const char *src, size_t n);

/*
 * Emulates an invocation of snprintf with format string "%s%zu$". Used for the rounds string.
 */
__host__ __device__ int local_snprintf_rounds(char *dst, size_t n, char const *prefix, size_t rounds);
