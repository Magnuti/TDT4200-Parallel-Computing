#include "local-string.cuh"

__host__ __device__ bool local_strneq(const char *str1, const char *str2, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (str1[i] != str2[i]) {
            return false;
        }
        if (str1[i] == '\0') {
            break;
        }
    }
    return true;
}

#ifdef SHACRYPT_UTIL_TEST_STRNEQ
#include <stdio.h>
int main() {
    char const str1[] = "abc123";
    char const str2[] = "abc";
    int i = local_strneq(str1, str2, 3);
    printf("Strings equal: %d\n", i);
    return 0;
}
#endif

__host__ __device__ unsigned long int local_strtoul10(const char* str, char** endptr) {
    unsigned long int number = 0;
    const char *iter = str;
    // Iterate through every character of the number
    while (true) {
        // Check if number character
        if (*iter < '0' || *iter > '9') {
            break;
        }

        // Add new digit to number
        unsigned long int index_number = *iter - '0';
        number = number * 10 + index_number;

        iter++;
    }

    *endptr = (char *) iter;
    return number;
}

#ifdef SHACRYPT_UTIL_TEST_STRTOUL10
#include <stdio.h>
int main() {
    char const str[] = "10506hello";
    char *endptr;
    unsigned long int i = local_strtoul10(str, &endptr);
    printf("Number in string: %ld\n", i);
    printf("Remaining string: %s\n", endptr);
    return 0;
}
#endif

__host__ __device__ size_t local_strlen(const char *str) {
    for (size_t i = 0; ; i++) {
        if (str[i] == '\0') {
            return i;
        }
    }
}

#ifdef SHACRYPT_UTIL_TEST_STRLEN
#include <stdio.h>
__host__ __device__ int main() {
    char str[] = "fcba73";
    int i = local_strlen(str);
    printf("Length of string: %d\n", i);
    return 0;
}
#endif

__host__ __device__ size_t local_strnlen(const char *str, size_t n) {
    for (size_t i = 0; ; i++) {
        if (i >= n) {
            return i;
        }
        if (str[i] == '\0') {
            return i;
        }
    }
}

#ifdef SHACRYPT_UTIL_TEST_STRNLEN
#include <stdio.h>
int main() {
    char str[] = "fcba73";
    int i = local_strnlen(str, 4);
    printf("Length of string with limited length: %d\n", i);
    return 0;
}
#endif

__host__ __device__ size_t local_strcspn(const char *str1, const char *str2) {
    // Scan first string
    for (size_t i = 0; ; i++) {
        // End of first string, return length
        if (str1[i] == '\0') {
            return i;
        }
        // Scan str2 for match for current str1 character
        for (size_t j = 0; ; j++) {
            // End of str2, no match found
            if (str2[j] == '\0') {
                break;
            }
            // Compare char for match
            if (str1[i] == str2[j]) {
                return i;
            }
        }
    }
}

#ifdef SHACRYPT_UTIL_TEST_STRCSPN
#include <stdio.h>
int main() {
    char str[] = "fcba73";
    char keys[] = "1234567890";
    int i = local_strcspn(str, keys);
    printf("Length of string before first number (if any): %d\n", i);
    return 0;
}
#endif

__host__ __device__ char *local_strncpy(char *dst, const char *src, size_t n) {
    bool src_valid = true;
    for (size_t i = 0; i < n; i++) {
        if (src_valid && src[i] == '\0') {
            src_valid = false;
        }
        dst[i] = src_valid ? src[i] : '\0';
    }
    return dst;
}

#ifdef SHACRYPT_UTIL_TEST_STRNCPY
#include <stdio.h>
int main() {
    char src[] = "abcdef";
    char dst[64];
    local_strncpy(dst, src, 7);
    printf("source=\"%s\" destination=\"%s\"\n", src, dst);
    return 0;
}
#endif

__host__ __device__ char *local_stpncpy(char *dst, const char *src, size_t n) {
    size_t size = local_strnlen(src, n);
    for (size_t i = 0; i < size; i++) {
        *dst = src[i];
        dst++;
    }
    if (size < n) {
        *dst = '\0';
    }
    return dst;
}

#ifdef SHACRYPT_UTIL_TEST_STPNCPY
#include <stdio.h>
int main() {
    char src[] = "hello1234";
    char dst[64];
    char *end = local_stpncpy(dst, src, 10);
    printf("src=\"%s\" dst=\"%s\" end_pos=%ld\n", src, dst, (end - dst));
    return 0;
}
#endif

__host__ __device__ int local_snprintf_rounds(char *dst, size_t n, char const *prefix, size_t rounds) {
    // Format: %s%zu$

    size_t dst_index = 0;

    // Add prefix
    for (size_t i = 0; prefix[i] != '\0'; i++) {
        // Write only if available space
        if (dst_index < n - 1) {
            dst[dst_index] = prefix[i];
        }
        dst_index++;
    }

    // Find length of number
    size_t rounds_length = 1;
    size_t rounds_tmp = rounds;
    while (rounds_tmp >= 9) {
        rounds_tmp /= 10;
        rounds_length++;
    }

    // Add number
    size_t rounds_suffix = rounds;
    for(size_t i = 0; i < rounds_length; i++) {
        size_t digit = rounds_suffix;
        size_t digit_magnitude = 1;

        // Find first digit
        for(size_t j = i; j < rounds_length - 1; j++) {
            digit /= 10;
            digit_magnitude *= 10;
        }

        // Remove first digit
        rounds_suffix -= digit * digit_magnitude;

        // Write only if available space
        if (dst_index < n - 1) {
            dst[dst_index] = '0' + digit;
        }
        dst_index++;
    }

    // Add $
    // Write only if available space
    if (dst_index < n - 1) {
        dst[dst_index] = '$';
    }
    dst_index++;

    // Add zero-terminator
    if (n > 0) {
        dst[MIN(dst_index, n - 1)] = '\0';
    }

    // Return length (minus null-terminator) that would have been written,
    // ignoring n and how many were actually written
    return dst_index;
}

#ifdef SHACRYPT_UTIL_TEST_SNPRINTF_ROUNDS
#include <stdio.h>
int main() {
    char prefix[] = "helloworld";
    size_t rounds = 5000;
    char dst[64];
    int length;
    length = snprintf(dst, 64, "%s%zu$", prefix, rounds);
    printf("expected=\"%s\" length=\"%d\"\n", dst, length);
    length = local_snprintf_rounds(dst, 64, prefix, rounds);
    printf("actual__=\"%s\" length=\"%d\"\n", dst, length);
    return 0;
}
#endif
