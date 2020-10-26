#pragma once

#define MAX_PASSWORD_LENGTH 127
#define MAX_SALT_LENGTH 63

__device__ char *sha512_crypt_r(const char *key, const char *salt, char *buffer, int buflen);
