#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "common.hpp"

typedef struct {
    // Don't print normal stuff
    bool quiet;
    // Print all passwords attempted
    bool verbose;
    // Path to shadow file
    char shadow_file[MAX_FILE_PATH_LENGTH + 1];
    // Path to output file
    char result_file[MAX_FILE_PATH_LENGTH + 1];
    // Path to dictionary of symbols
    char dict_file[MAX_FILE_PATH_LENGTH + 1];
    // Max number of symbols to repeat
    size_t max_length;
    // Max number of symbols to repeat
    char separator[MAX_SEPARATOR_LENGTH + 1];
} Options;

typedef enum {
    ALG_UNKNOWN = 0,
    ALG_MD5,
    ALG_SHA256,
    ALG_SHA512,
} CryptAlg;

typedef struct {
    // User field
    char user[MAX_USER_LENGTH + 1];
    // "Encrypted password" field, including salt, parameters and password hash
    char passfield[MAX_PASSFIELD_LENGTH + 1];
    // Hash algorithm
    CryptAlg alg;
    // Exclusive offset of the "salt" end
    size_t salt_end;
} ShadowEntry;

typedef struct {
    char (*elements)[MAX_DICT_ELEMENT_LENGTH + 1];
    size_t length;
    size_t max_length;
} Dictionary;

typedef enum {
    ACTION_WORK = 0,
    ACTION_WAIT,
} CrackJobAction;

typedef struct {
    char passfield[MAX_PASSFIELD_LENGTH + 1];
    CryptAlg alg;
    size_t salt_end;
    char probe[MAX_PASSWORD_LENGTH + 1];
    CrackJobAction action;
} CrackJob;

typedef enum {
    STATUS_PENDING = 0,
    STATUS_SKIP,
    STATUS_SUCCESS,
    STATUS_FAIL,
    STATUS_ERROR,
} CrackResultStatus;

typedef struct {
    CrackResultStatus status;
    char password[MAX_PASSWORD_LENGTH + 1];
} CrackResult;

typedef struct {
    CrackResultStatus status;
    char password[MAX_PASSWORD_LENGTH + 1];
    double duration;
    long attempts;
    char user[MAX_USER_LENGTH + 1];
    char passfield[MAX_PASSFIELD_LENGTH + 1];
    CryptAlg alg;
} ExtendedCrackResult;

typedef struct {
    size_t statuses[STATUS_ERROR + 1];
    double duration;
    long attempts;
} OverviewCrackResult;

typedef struct {
    char probe[MAX_PASSWORD_LENGTH + 1];
    size_t size;
    size_t *dict_positions;
    char (*symbols)[MAX_DICT_ELEMENT_LENGTH + 1];
} ProbeConfig;

/*
 * Parses CLI arguments into the provided options. Returns true if success.
 */
bool parse_cli_args(Options *options, int argc, char **argv);

/*
 * Opens the file with the given mode (see fopen(3)). Returns true if success.
 */
bool open_file(FILE **file, char const *path, char const *mode);

/*
 * Reads a dictionary file. The dictionary element array will be automatically allocated. Returns true if success.
 */
bool read_dictionary(Dictionary *dict, Options *options, char const *path);

/*
 * Reads a dictionary file. The dictionary element array will be automatically allocated. Returns true if success.
 */
bool get_next_shadow_entry(ShadowEntry *entry, FILE *file);

/*
 * Get string representation of a crypt algorithm.
 */
char const *cryptalg_to_string(CryptAlg alg);

/*
 * Get string representation of a cracking result status.
 */
char const *crack_result_status_to_string(CrackResultStatus alg);
