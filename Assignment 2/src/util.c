#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <getopt.h>

#include "util.h"

static bool read_cli_arg_string(char *destination, size_t max_length, char const *optarg);
static bool read_cli_arg_int(long *destination, char const *optarg);
static void parse_shadow_passfield(ShadowEntry *entry);

bool parse_cli_args(Options *options, int argc, char **argv) {
    char usage[128];
    snprintf(usage, 128, "Usage: %s [-q] [-v] -i<shadow_file> -d<dictionary_file> [-l<max_length>] [-s<separator>] [-o<result_file>]", argv[0]);

    // Set defaults
    // Empty paths
    options->shadow_file[0] = 0;
    options->result_file[0] = 0;
    options->dict_file[0] = 0;
    // 1 symbol
    options->max_length = 1;
    // Empty string
    options->separator[0] = 0;

    // Read args from command line
    char opt;
    size_t opt_count = 0;
    while ((opt = getopt(argc, argv, "qvi:o:d:l:s:")) != -1) {
        long tmp_arg_num;
        switch (opt) {
        case 'q':
            options->quiet = true;
            break;
        case 'v':
            options->verbose = true;
            break;
        case 'i':
            if (!read_cli_arg_string(options->shadow_file, MAX_FILE_PATH_LENGTH, optarg)) {
                fprintf(stderr, "Shadow file path too long.\n\n%s\n", usage);
                return false;
            }
            break;
        case 'o':
            if (!read_cli_arg_string(options->result_file, MAX_FILE_PATH_LENGTH, optarg)) {
                fprintf(stderr, "Result file path too long.\n\n%s\n", usage);
                return false;
            }
            break;
        case 'd':
            if (!read_cli_arg_string(options->dict_file, MAX_FILE_PATH_LENGTH, optarg)) {
                fprintf(stderr, "Dictionary file path too long.\n\n%s\n", usage);
                return false;
            }
            break;
        case 'l':
            if (!read_cli_arg_int(&tmp_arg_num, optarg) || tmp_arg_num < 1) {
                fprintf(stderr, "Invalid max length.\n\n%s\n", usage);
                return false;
            }
            options->max_length = (size_t) tmp_arg_num;
            break;
        case 's':
            if (!read_cli_arg_string(options->separator, MAX_SEPARATOR_LENGTH, optarg)) {
                fprintf(stderr, "Separator too long.\n\n%s\n", usage);
                return false;
            }
            break;
        default:
            fprintf(stderr, "Unknown argument.\n\n%s\n", usage);
            return false;
        }
        opt_count++;
    }

    // Check if required args are set
    if (options->shadow_file[0] == 0) {
        fprintf(stderr, "Shadow file not provided.\n\n%s\n", usage);
        return false;
    }
    if (options->dict_file[0] == 0) {
        fprintf(stderr, "Dictionary file not provided.\n\n%s\n", usage);
        return false;
    }

    return true;
}

static bool read_cli_arg_string(char *destination, size_t max_length, char const *optarg) {
    // optarg shadows global optarg
    // "max_length + 1" instead of max_length to allow checking if there's too many characters
    size_t optarg_length = strnlen(optarg, max_length + 1);
    if (optarg_length > max_length) {
        return false;
    }
    strncpy(destination, optarg, optarg_length + 1);
    return true;
}

static bool read_cli_arg_int(long *destination, char const *optarg) {
    // optarg shadows global optarg
    // Reset errno
    errno = 0;
    char *tmp_end;
    *destination = strtol(optarg, &tmp_end, 10);
    if (tmp_end[0] != 0 || tmp_end == optarg || errno) {
        return false;
    }
    return true;
}

bool open_file(FILE **file, char const *path, char const *mode) {
    errno = 0;
    *file = fopen(path, mode);
    if (*file == NULL) {
        fprintf(stderr, "Failed to open file \"%s\".\n", path);
        perror("fopen");
        return false;
    }
    return true;
}

bool read_dictionary(Dictionary *dict, Options *options, char const *path) {
    static const size_t INITIAL_ELEMENTS = 128;

    FILE *file;
    if (!open_file(&file, path, "r")) {
        return false;
    }

    // Initialize dict
    dict->length = 0;
    dict->max_length = INITIAL_ELEMENTS;
    dict->elements = malloc(dict->max_length * (MAX_DICT_ELEMENT_LENGTH + 1));

    char *line = NULL;
    size_t line_length = 0;
    while (getline(&line, &line_length, file) != -1) {
        // Expand array if necessary
        if (dict->length == dict->max_length) {
            char (*old_elements)[] = dict->elements;
            size_t old_max_length = dict->max_length;
            dict->max_length = old_max_length * 2;
            dict->elements = malloc(dict->max_length * (MAX_DICT_ELEMENT_LENGTH + 1));
            if (dict->elements == NULL) {
                fprintf(stderr, "Failed to allocate memory for dictionary.\n");
                perror("malloc");
                return false;
            }
            memcpy(dict->elements, old_elements, old_max_length * (MAX_DICT_ELEMENT_LENGTH + 1));
            free(old_elements);
        }

        // Strip newline from string (makes line_length wrong)
        line[strcspn(line, "\r\n")] = 0;
        // Copy line into dictionary
        char *dict_element = dict->elements[dict->length];
        strncpy(dict_element, line, MAX_DICT_ELEMENT_LENGTH);
        dict_element[MAX_DICT_ELEMENT_LENGTH] = 0;
        dict->length++;
    }
    free(line);

    fclose(file);

    if (!options->quiet) {
        printf("Read %ld words from dictionary file.\n", dict->length);
    }

    return true;
}

bool get_next_shadow_entry(ShadowEntry *entry, FILE *file) {
    // Initialize entry
    memset(entry, 0, sizeof(ShadowEntry));

    char *line = NULL;
    size_t line_length = 0;

    // Read line and check if EOF
    if (getline(&line, &line_length, file) == -1) {
        return false;
    }

    // Strip newline from string (makes line_length wrong)
    line[strcspn(line, "\r\n")] = 0;

    // Parse user and password fields from shadow line
    char *line_buffer = line;
    for (size_t i = 0; i < 2; i++) {
        char *token = strsep(&line_buffer, ":");
        if (token == NULL) {
            break;
        }
        switch (i) {
        case 0:
            strncpy(entry->user, token, MAX_SHADOW_USER_LENGTH);
            entry->user[MAX_SHADOW_USER_LENGTH] = 0;
            break;
        case 1:
            strncpy(entry->passfield, token, MAX_SHADOW_PASSFIELD_LENGTH);
            entry->passfield[MAX_SHADOW_PASSFIELD_LENGTH] = 0;
            break;
        default:
            break;
        }
    }

    free(line);

    // Parse passfield
    parse_shadow_passfield(entry);

    return true;
}

static void parse_shadow_passfield(ShadowEntry *entry) {
    // Decide alg.
    if (strncmp(entry->passfield, "$1$", 3) == 0) {
        entry->alg = ALG_MD5;
    }
    else if (strncmp(entry->passfield, "$5$", 3) == 0) {
        entry->alg = ALG_SHA256;
    }
    else if (strncmp(entry->passfield, "$6$", 3) == 0) {
        entry->alg = ALG_SHA512;
    }

    // Last field is the hash
    size_t passfield_length = strnlen(entry->passfield, MAX_SHADOW_PASSFIELD_LENGTH);
    for (size_t i; i < passfield_length; i++) {
        if (entry->passfield[i] == '$') {
            entry->salt_end = i;
        }
    }
}

char const *cryptalg_to_string(CryptAlg alg) {
    switch (alg) {
    case ALG_MD5:
        return "MD5";
    case ALG_SHA256:
        return "SHA256";
    case ALG_SHA512:
        return "SHA512";
    default:
        return "UNKNOWN";
    }
}

char const *crack_result_status_to_string(CrackResultStatus status) {
    switch (status) {
    case STATUS_PENDING:
        return "PENDING";
    case STATUS_SKIP:
        return "SKIP";
    case STATUS_SUCCESS:
        return "SUCCESS";
    case STATUS_FAIL:
        return "FAIL";
    default:
        return "UNKNOWN";
    }
}
