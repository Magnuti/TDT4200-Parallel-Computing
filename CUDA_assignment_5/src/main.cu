#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common.hpp"
#include "util.hpp"
#include "lib/local-string.cuh"
#include "lib/sha512crypt.cuh"

static bool init(Options *options, Dictionary *dict, FILE **shadow_file, FILE **result_file, int argc, char **argv);
static bool init_cuda();
static void cleanup(FILE *shadow_file, FILE *result_file, Dictionary *dict);
static void crack(ExtendedCrackResult *result, Options *options, Dictionary *dict, ShadowEntry *entry);
static bool prepare_job(CrackJob *job, ShadowEntry *entry, ProbeConfig *config, Options *options, Dictionary *dict);
static bool get_next_probe(ProbeConfig *config, Options *options, Dictionary *dict);
static void handle_result(Options *options, ExtendedCrackResult *result, OverviewCrackResult *overview_result, FILE *result_file);
static void handle_overview_result(Options *options, OverviewCrackResult *overview_result);
static void crack_job(CrackResult results[], CrackJob jobs[]);

/*
 * Main entrypoint.
 */
int main(int argc, char **argv) {
    Options options = {};
    Dictionary dict = {};
    FILE *shadow_file = NULL;
    FILE *result_file = NULL;

    if (!init(&options, &dict, &shadow_file, &result_file, argc, argv)) {
        cleanup(shadow_file, result_file, &dict);
        return EXIT_FAILURE;
    }

    // Iterate and crack shadow entries
    if (!options.quiet) {
        printf("\nEntries:\n");
    }
    OverviewCrackResult overview_result = {};
    ShadowEntry shadow_entry;
    while (get_next_shadow_entry(&shadow_entry, shadow_file)) {
        ExtendedCrackResult result;
        crack(&result, &options, &dict, &shadow_entry);
        if (result.status == STATUS_ERROR) {
            fprintf(stderr, "Stopping due to an error.\n");
            break;
        }
        handle_result(&options, &result, &overview_result, result_file);
    }

    // Handle overall result
    handle_overview_result(&options, &overview_result);

    cleanup(shadow_file, result_file, &dict);
    return EXIT_SUCCESS;
}

/*
 * Initialize general stuff.
 */
static bool init(Options *options, Dictionary *dict, FILE **shadow_file, FILE **result_file, int argc, char **argv) {
    // Parse CLI args
    if (!parse_cli_args(options, argc, argv)) {
        return false;
    }

    // Print some useful info
    if (!options->quiet) {
        printf("Chosen CUDA grid size: %d\n", GRID_SIZE);
        printf("Chosen CUDA block size: %d\n", BLOCK_SIZE);
        printf("Max symbols: %ld\n", options->max_length);
        printf("Symbol separator: \"%s\"\n", options->separator);
    }

    // Open shadow file
    if (!options->quiet) {
        printf("Shadow file: %s\n", options->shadow_file);
    }
    if (!open_file(shadow_file, options->shadow_file, "r")) {
        return false;
    }
    // Open output file if provided
    if (options->result_file[0] != 0) {
        if (!options->quiet) {
            printf("Output file: %s\n", options->result_file);
        }
        if (!open_file(result_file, options->result_file, "w")) {
            return false;
        }
    }
    // Read full directory
    if (!options->quiet) {
        printf("Dictionary file: %s\n", options->dict_file);
    }
    if (!read_dictionary(dict, options, options->dict_file)) {
        return false;
    }

    // Init CUDA
    if (!init_cuda()) {
        return false;
    }

    return true;
}

/*
 * Initialize CUDA stuff.
 */
static bool init_cuda() {
    // Make sure at least one CUDA-capable device exists
    int device_count;
    // TODO get device count
    printf("CUDA device count: %d\n", device_count);
    if (device_count < 1) {
        fprintf(stderr, "No CUDA devices present.\n");
        return false;
    }

    // Print some useful info
    cudaDeviceProp prop;
    // TODO get device properties
    printf("CUDA device #0:\n");
    printf("\tName: %s\n", prop.name);
    printf("\tCompute capability: %d.%d\n", prop.major, prop.minor);
    printf("\tMultiprocessors: %d\n", prop.multiProcessorCount);
    printf("\tWarp size: %d\n", prop.warpSize);
    printf("\tGlobal memory: %.1fGiB bytes\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("\tPer-block shared memory: %.1fkiB\n", prop.sharedMemPerBlock / 1024.0);
    printf("\tPer-block registers: %d\n", prop.regsPerBlock);

    // Check for any previous errors
    cudaError_t error = cudaPeekAtLastError();
    if (error) {
        fprintf(stderr, "A CUDA error has occurred while initializing: %s\n", cudaGetErrorString(error));
        return false;
    }

    return true;
}

/*
 * Cleanup stuff.
 */
static void cleanup(FILE *shadow_file, FILE *result_file, Dictionary *dict) {
    if (shadow_file) {
        fclose(shadow_file);
    }
    if (result_file) {
        fclose(result_file);
    }
    if (dict->elements) {
        free(dict->elements);
    }
}

/*
 * Crack a shadow password entry.
 */
static void crack(ExtendedCrackResult *total_result, Options *options, Dictionary *dict, ShadowEntry *entry) {
    // Initialize main result early in case of early return
    memset(total_result, 0, sizeof(ExtendedCrackResult));
    strncpy(total_result->user, entry->user, MAX_USER_LENGTH);
    strncpy(total_result->passfield, entry->passfield, MAX_PASSFIELD_LENGTH);
    total_result->alg = entry->alg;

    // Skip if not SHA512
    if (entry->alg != ALG_SHA512) {
        total_result->status = STATUS_SKIP;
        return;
    }

    ProbeConfig config = {};
    config.dict_positions = (size_t *) malloc(options->max_length * sizeof(size_t));
    config.symbols = (char (*)[MAX_DICT_ELEMENT_LENGTH + 1]) malloc(options->max_length * (MAX_DICT_ELEMENT_LENGTH + 1) * sizeof(char));

    // TODO allocate host and device arrays for CrackJobs and CrackResults

    // Start time measurement
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Try probes until the status changes (when a match is found or the search space is exhausted)
    while(total_result->status == STATUS_PENDING) {
        // Prepare new jobs
        bool more_probes = true;
        // TODO fill host job array with new jobs
        for (size_t i = 0; i < ???; i++) {
            if (!prepare_job(&???[i], entry, &config, options, dict)) {
                more_probes = false;
                break;
            }
        }

        // Copy jobs to device
        // TODO

        // Start kernel
        // TODO

        // Copy results from device
        // TODO

        // Check for error
        cudaError_t error = cudaPeekAtLastError();
        if (error) {
            fprintf(stderr, "A CUDA error has occurred while cracking: %s\n", cudaGetErrorString(error));
            total_result->status = STATUS_ERROR;
            break;
        }

        // Handle results
        for (size_t i = 0; i < ???; i++) {
            CrackResult *result = &???[i];

            // Skip if skip
            if (total_result->status == STATUS_SKIP) {
                continue;
            }

            // Keep track of probes tested
            total_result->attempts++;

            // Accept if success (currently the only one it makes sense to stop on)
            if (result->status == STATUS_SUCCESS) {
                total_result->status = result->status;
                strncpy(total_result->password, result->password, MAX_PASSWORD_LENGTH);
                // Ignore all job results after this one
                break;
            }
        }

        // Check if search space is exhausted and not match has been found
        if (!more_probes && total_result->status == STATUS_PENDING) {
            total_result->status = STATUS_FAIL;
        }
    }

    // End time measurement and record duration
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    total_result->duration = ((double) (end_time.tv_sec - start_time.tv_sec)) + ((double) (end_time.tv_nsec - start_time.tv_nsec)) * 1e-9;

    // Cleanup
    free(config.dict_positions);
    free(config.symbols);
    // TODO free new host and device arrays
}

static bool prepare_job(CrackJob *job, ShadowEntry *entry, ProbeConfig *config, Options *options, Dictionary *dict) {
    // Zeroize
    memset(job, 0, sizeof(CrackJob));

    bool more_probes = get_next_probe(config, options, dict);
    if (more_probes) {
        job->action = ACTION_WORK;
        strncpy(job->passfield, entry->passfield, MAX_PASSWORD_LENGTH);
        job->alg = entry->alg;
        job->salt_end = entry->salt_end;
        strncpy(job->probe, config->probe, MAX_PASSWORD_LENGTH);
    } else {
        job->action = ACTION_WAIT;
    }

    if (options->verbose) {
        printf("%s\n", job->probe);
    }

    return more_probes;
}

/*
 * Build the next probe. Returns false with an empty probe when the search space is exhausted.
 */
static bool get_next_probe(ProbeConfig *config, Options *options, Dictionary *dict) {
    // Check if dict is empty
    if (dict->length == 0) {
        return false;
    }

    // Find last symbol which can be replaced with the next one, if any exists
    ssize_t last_replaceable_pos = -1;
    for (size_t i = 0; i < config->size; i++) {
        if (config->dict_positions[i] < dict->length - 1) {
            last_replaceable_pos = i;
        }
    }

    // A symbol can be replaced, replace last one and reset all behind it
    if (last_replaceable_pos >= 0) {
        size_t new_dict_pos = config->dict_positions[last_replaceable_pos] + 1;
        config->dict_positions[last_replaceable_pos] = new_dict_pos;
        strncpy(config->symbols[last_replaceable_pos], dict->elements[new_dict_pos], MAX_DICT_ELEMENT_LENGTH);
        for (size_t i = last_replaceable_pos + 1; i < config->size; i++) {
            config->dict_positions[i] = 0;
            strncpy(config->symbols[i], dict->elements[0], MAX_DICT_ELEMENT_LENGTH);
        }
    }
    // No symbols can be replaced and no more symbols are allowed, return error
    else if (config->size == options->max_length) {
        config->probe[0] = 0;
        return false;
    }
    // New symbol can be added, reset all previous positions and add it
    else {
        config->size++;
        for (size_t i = 0; i < config->size; i++) {
            config->dict_positions[i] = 0;
            strncpy(config->symbols[i], dict->elements[0], MAX_DICT_ELEMENT_LENGTH);
        }
    }

    // Build probe
    config->probe[0] = 0;
    for (size_t i = 0; i < config->size; i++) {
        if (i > 0) {
            strncat(config->probe, options->separator, MAX_PASSWORD_LENGTH);
        }
        strncat(config->probe, config->symbols[i], MAX_PASSWORD_LENGTH);
    }

    return true;
}

/*
 * Handle result from trying to crack a single password.
 */
static void handle_result(Options *options, ExtendedCrackResult *result, OverviewCrackResult *overview_result, FILE *result_file) {
    // Make representations
    char const *alg_str = cryptalg_to_string(result->alg);
    char const *status_str = crack_result_status_to_string(result->status);
    double attempts_per_second = result->attempts / result->duration;

    // Format and print
    size_t const static max_output_length = 1023;
    char *output = (char *) malloc(max_output_length + 1);
    snprintf(output, max_output_length + 1, "user=\"%s\", alg=\"%s\" status=\"%s\" duration=\"%fs\" attempts=\"%ld\" attempts_per_second=\"%f\" password=\"%s\"",
            result->user, alg_str, status_str, result->duration, result->attempts, attempts_per_second, result->password);
    if (!options->quiet) {
        printf("%s\n", output);
    }
    if (result_file) {
        fprintf(result_file, "%s\n", output);
        fflush(result_file);
    }
    free(output);

    // Update overview
    overview_result->statuses[result->status]++;
    overview_result->duration += result->duration;
    overview_result->attempts += result->attempts;
}

/*
 * Handle result from trying to crack all passwords.
 */
static void handle_overview_result(Options *options, OverviewCrackResult *result) {
    if (!options->quiet) {
        printf("\nOverview:\n");
        printf("Total duration: %.3fs\n", result->duration);
        printf("Total attempts: %ld\n", result->attempts);
        printf("Total attempts per second: %.3f\n", result->attempts / result->duration);
        printf("Skipped: %ld\n", result->statuses[STATUS_SKIP]);
        printf("Successful: %ld\n", result->statuses[STATUS_SUCCESS]);
        printf("Failed: %ld\n", result->statuses[STATUS_FAIL]);
    }
}

/*
 * Hash probe and compare.
 */
static void crack_job(CrackResult results[], CrackJob jobs[]) {
    // TODO set using unique index into arrays
    CrackResult *result = ???;
    CrackJob *job = ???;

    // Zeroize result
    result->status = STATUS_PENDING;
    result->password[0] = 0;

    // Nothing to do here
    if (job->action == ACTION_WAIT) {
        result->status = STATUS_SKIP;
        return;
    }

    // Only accept SHA512 (redundant check)
    if (job->alg != ALG_SHA512) {
        result->status = STATUS_SKIP;
        return;
    }

    // Copy probe into shared memory
    char *probe = job->probe;

    // Copy salt part of passfield into shared memory (same value is used by all threads)
    char salt[MAX_SALT_LENGTH];
    local_strncpy(salt, job->passfield, MIN(job->salt_end, MAX_SALT_LENGTH));
    salt[job->salt_end] = 0;

    // Call sha512_crypt_r directly using register buffer
    char new_passfield[MAX_PASSFIELD_LENGTH + 1];
    sha512_crypt_r(probe, salt, new_passfield, MAX_PASSFIELD_LENGTH + 1);
    if (new_passfield != NULL && local_strneq(job->passfield, new_passfield, MAX_PASSFIELD_LENGTH)) {
        // Match found, abort search
        result->status = STATUS_SUCCESS;
        local_strncpy(result->password, probe, MAX_PASSWORD_LENGTH);
    }
}
