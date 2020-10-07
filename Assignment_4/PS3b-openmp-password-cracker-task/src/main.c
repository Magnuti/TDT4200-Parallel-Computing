#define _GNU_SOURCE
#include <crypt.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "util.h"

static bool init(Options * options, Dictionary * dict, FILE ** shadow_file,
		 FILE ** result_file, int argc, char **argv);
static void cleanup(FILE * shadow_file, FILE * result_file,
		    Dictionary * dict);
static void crack(ExtendedCrackResult * result, Options * options,
		  Dictionary * dict, ShadowEntry * entry);
static void crack_job(CrackResult * results, CrackJob * jobs);
static bool get_next_probe(ProbeConfig * config, Options * options,
			   Dictionary * dict);
static void handle_result(Options * options, ExtendedCrackResult * result,
			  OverviewCrackResult * overview_result,
			  FILE * result_file);
static void handle_overview_result(Options * options,
				   OverviewCrackResult * overview_result);
static bool prepare_job(CrackJob * job, ShadowEntry * entry,
			ProbeConfig * config, Options * options,
			Dictionary * dict);

/*
 * Main entrypoint.
 */
int main(int argc, char **argv)
{
    Options options = { };
    Dictionary dict = { };
    FILE *shadow_file = NULL;
    FILE *result_file = NULL;



    if (!init(&options, &dict, &shadow_file, &result_file, argc, argv)) {
	cleanup(shadow_file, result_file, &dict);
	return EXIT_FAILURE;
    }
    // If init successful, try to crack all shadow entries
    if (!options.quiet) {
	printf("\nEntries:\n");
    }
    OverviewCrackResult overview_result = { };
    ShadowEntry shadow_entry;
    while (get_next_shadow_entry(&shadow_entry, shadow_file)) {
	ExtendedCrackResult result;
	crack(&result, &options, &dict, &shadow_entry);
	handle_result(&options, &result, &overview_result, result_file);
    }

    // Handle overall result
    handle_overview_result(&options, &overview_result);

    cleanup(shadow_file, result_file, &dict);
    return EXIT_SUCCESS;
}


/*
 * Initialize master stuff.
 */
static bool
init(Options * options, Dictionary * dict, FILE ** shadow_file,
     FILE ** result_file, int argc, char **argv)
{
    // Parse CLI args
    if (!parse_cli_args(options, argc, argv)) {
	return false;
    }
    // Print some useful info
    if (!options->quiet) {
	printf("Threads: %d\n", omp_get_max_threads());
	printf("Max symbols: %ld\n", options->max_length);
	printf("Symbol separator: \"%s\"\n", options->separator);
    }
    // Open shadow file
    if (!options->quiet) {
	printf("Shadow file: %s\n", options->shadow_file);
    }
    if (!open_file(shadow_file, options->shadow_file, "r")) {
	cleanup(*shadow_file, *result_file, dict);
	return false;
    }
    // Open output file if provided
    if (options->result_file[0] != 0) {
	if (!options->quiet) {
	    printf("Output file: %s\n", options->result_file);
	}
	if (!open_file(result_file, options->result_file, "w")) {
	    cleanup(*shadow_file, *result_file, dict);
	    return false;
	}
    }
    // Read full directory
    if (!options->quiet) {
	printf("Dictionary file: %s\n", options->dict_file);
    }
    if (!read_dictionary(dict, options, options->dict_file)) {
	cleanup(*shadow_file, *result_file, dict);
	return false;
    }

    return true;
}

/*
 * Cleanup master stuff.
 */
static void
cleanup(FILE * shadow_file, FILE * result_file, Dictionary * dict)
{
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
 * Crack a shadow password entry as master.
 */
static void
crack(ExtendedCrackResult * result, Options * options, Dictionary * dict,
      ShadowEntry * entry)
{
    // Initialize result
    memset(result, 0, sizeof(ExtendedCrackResult));
    strncpy(result->user, entry->user, MAX_SHADOW_USER_LENGTH);
    strncpy(result->passfield, entry->passfield,
	    MAX_SHADOW_PASSFIELD_LENGTH);
    result->alg = entry->alg;

    // Accept only known algs
    if (entry->alg == ALG_UNKNOWN) {
	result->status = STATUS_SKIP;
	return;
    }
    // Setup vars for cracking
    ProbeConfig config = { };
    config.dict_positions = calloc(options->max_length, sizeof(size_t));
    config.symbols =
	calloc(options->max_length, MAX_DICT_ELEMENT_LENGTH + 1);

    // TODO: Get maximum number of threads
    size_t num_threads = 1;

    CrackJob *jobs = calloc(num_threads, sizeof(CrackJob));
    for (size_t i = 0; i < num_threads; i++) {
	strncpy(jobs[i].passfield, entry->passfield,
		MAX_SHADOW_PASSFIELD_LENGTH);
    }
    CrackResult *results = calloc(num_threads, sizeof(CrackResult));

    // Start time measurement
    // TODO: get omp walltime
    double start_time = 0;

    // Try probes until the status changes (when a match is found or the search space is exhausted)
    while (result->status == STATUS_PENDING) {
	// Make jobs with new probes
	bool more_probes = true;
	for (size_t i = 0; i < num_threads; i++) {
	    if (!prepare_job(&jobs[i], entry, &config, options, dict)) {
		more_probes = false;
		break;
	    }
	}

	// TODO: Parallelize using omp
#pragma omp single
	{
	    // TODO: Get thread index
	    size_t tid = 0;
	    crack_job(&results[tid], &jobs[tid]);
	}

	// Handle results
	for (size_t i = 0; i < num_threads; i++) {

	    // Skip if skip
	    if (results[i].status == STATUS_SKIP) {
		continue;
	    }
	    // Keep track of probes tested
	    result->attempts++;

	    // Accept if success (currently the only one it makes sense to stop on)
	    if (results[i].status != STATUS_PENDING) {
		result->status = results[i].status;
		strncpy(result->password, results[i].password,
			MAX_PASSWORD_LENGTH);
		// Ignore all job results after this one
		break;
	    }
	}

	// Check if search space is exhausted and not match has been found
	if (!more_probes && result->status == STATUS_PENDING) {
	    result->status = STATUS_FAIL;
	}
    }

    // End time measurement
    // TODO: get omp walltime
    double end_time = 0;
    result->duration = end_time - start_time;

    free(config.dict_positions);
    free(config.symbols);
    free(jobs);
    free(results);
}

static void crack_job(CrackResult * result, CrackJob * job)
{

    char const *new_passfield;

    // Zeroize result
    result->status = STATUS_PENDING;
    result->password[0] = 0;

    // Nothing to do here
    if (job->action == ACTION_WAIT) {
	result->status = STATUS_SKIP;
    }
    // Only accept known (redundant check)
    if (job->alg == ALG_UNKNOWN) {
	result->status = STATUS_SKIP;
    }

    if (result->status != STATUS_SKIP) {
	struct crypt_data data[1] = { 0 };
	new_passfield = crypt_r(job->probe, job->passfield, data);

	if (new_passfield != NULL
	    && strncmp(job->passfield, new_passfield,
		       MAX_SHADOW_PASSFIELD_LENGTH) == 0) {
	    // Match found, abort search
	    result->status = STATUS_SUCCESS;
	    strncpy(result->password, job->probe, MAX_PASSWORD_LENGTH);
	    result->password[MAX_PASSWORD_LENGTH] = 0;
	}
    }
}

/* 
 * Prepares job
 */
static bool
prepare_job(CrackJob * job, ShadowEntry * entry, ProbeConfig * config,
	    Options * options, Dictionary * dict)
{
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
static bool
get_next_probe(ProbeConfig * config, Options * options, Dictionary * dict)
{
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
	size_t new_dict_pos =
	    config->dict_positions[last_replaceable_pos] + 1;
	config->dict_positions[last_replaceable_pos] = new_dict_pos;
	strncpy(config->symbols[last_replaceable_pos],
		dict->elements[new_dict_pos], MAX_DICT_ELEMENT_LENGTH);
	for (size_t i = last_replaceable_pos + 1; i < config->size; i++) {
	    config->dict_positions[i] = 0;
	    strncpy(config->symbols[i], dict->elements[0],
		    MAX_DICT_ELEMENT_LENGTH);
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
	    strncpy(config->symbols[i], dict->elements[0],
		    MAX_DICT_ELEMENT_LENGTH);
	}
    }

    // Build probe
    config->probe[0] = 0;
    for (size_t i = 0; i < config->size; i++) {
	if (i > 0) {
	    strncat(config->probe, options->separator,
		    MAX_PASSWORD_LENGTH);
	}
	strncat(config->probe, config->symbols[i], MAX_PASSWORD_LENGTH);
    }

    return true;
}


/*
 * Handle result from trying to crack a single password.
 */
static void
handle_result(Options * options, ExtendedCrackResult * result,
	      OverviewCrackResult * overview_result, FILE * result_file)
{
    // Make representations
    char const *alg_str = cryptalg_to_string(result->alg);
    char const *status_str = crack_result_status_to_string(result->status);
    double attempts_per_second = result->attempts / result->duration;

    // Format and print
    size_t const static max_output_length = 1023;
    char *output = malloc(max_output_length + 1);
    snprintf(output, max_output_length + 1,
	     "user=\"%s\" alg=\"%s\" status=\"%s\" duration=\"%fs\" attempts=\"%ld\" attempts_per_second=\"%f\" password=\"%s\"",
	     result->user, alg_str, status_str, result->duration,
	     result->attempts, attempts_per_second, result->password);
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
static void
handle_overview_result(Options * options, OverviewCrackResult * result)
{
    if (!options->quiet) {
	printf("\nOverview:\n");
	printf("Total duration: %.3fs\n", result->duration);
	printf("Total attempts: %ld\n", result->attempts);
	printf("Total attempts per second: %.3f\n",
	       result->attempts / result->duration);
	printf("Skipped: %ld\n", result->statuses[STATUS_SKIP]);
	printf("Successful: %ld\n", result->statuses[STATUS_SUCCESS]);
	printf("Failed: %ld\n", result->statuses[STATUS_FAIL]);
    }
}
