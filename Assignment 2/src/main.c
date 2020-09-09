#include <crypt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <mpi.h>

#include "util.h"

static bool run_master(int argc, char **argv);
static bool init_master(Options *options, Dictionary *dict, FILE **shadow_file, FILE **result_file, int argc, char **argv);
static void master_cleanup(FILE *shadow_file, FILE *result_file, Dictionary *dict);
static void master_crack(ExtendedCrackResult *result, Options *options, Dictionary *dict, ShadowEntry *entry);
static void replica_crack(int rank, int number_of_processes);
static bool get_next_probe(ProbeConfig *config, Options *options, Dictionary *dict);
static void crack_job(CrackResult *result, CrackJob *job);
static void handle_result(Options *options, ExtendedCrackResult *result, OverviewCrackResult *overview_result, FILE *result_file);
static void handle_overview_result(Options *options, OverviewCrackResult *overview_result);

/*
 * Main entrypoint.
 */
int main(int argc, char **argv)
{
    int number_of_processes; // Number of processes
    int my_rank;             // The rank of this process

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0)
    {
        bool success = run_master(argc, argv);
        MPI_Finalize();
        return success ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    else
    {
        replica_crack(my_rank, number_of_processes);
        MPI_Finalize();
        return 0;
    }
}

/*
 * Entrypoint for master.
 */
static bool run_master(int argc, char **argv)
{
    Options options = {};
    Dictionary dict = {};
    FILE *shadow_file = NULL;
    FILE *result_file = NULL;

    bool init_success = init_master(&options, &dict, &shadow_file, &result_file, argc, argv);

    // If init successful, try to crack all shadow entries
    if (!options.quiet)
    {
        printf("\nEntries:\n");
    }
    OverviewCrackResult overview_result = {};
    if (init_success)
    {
        ShadowEntry shadow_entry;
        while (get_next_shadow_entry(&shadow_entry, shadow_file))
        {
            ExtendedCrackResult result;
            master_crack(&result, &options, &dict, &shadow_entry);
            handle_result(&options, &result, &overview_result, result_file);
        }

        // Send out ACTION_STOP to all replicas when there are no more shadow entires
        int mpi_size; // Number of processes
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        CrackJob *empty_crack_job = (CrackJob *)calloc(1, sizeof(CrackJob));
        CrackResult *emtpy_crack_result = (CrackResult *)calloc(1, sizeof(CrackResult));

        CrackJob stop_crack_job;
        stop_crack_job.action = ACTION_STOP;

        CrackJob *stop_jobs = (CrackJob *)calloc(mpi_size, sizeof(CrackJob));
        CrackResult *stop_replica_results = (CrackResult *)calloc(mpi_size, sizeof(CrackResult));

        for (int i = 1; i < mpi_size; i++)
        {
            stop_jobs[i] = stop_crack_job;
        }

        // Master receives the empty crack job, while replicas receive a CrackJob with ACTION_STOP

        MPI_Scatter(
            stop_jobs,        // The data we want to scatter
            sizeof(CrackJob), // How many MPI_BYTES to send --> one CrackJob
            MPI_BYTE,         // Send data type
            empty_crack_job,  // Buffer for receiving the emtpy CrackJob, i.e. only 0s
            sizeof(CrackJob), // How many MPI_BYTES to receive
            MPI_BYTE,         // Receive data type
            0,                // Root process
            MPI_COMM_WORLD    // Communicator
        );

        MPI_Gather(
            emtpy_crack_result,   // Data to send
            sizeof(CrackResult),  // How many MPI_BYTES to send
            MPI_BYTE,             // Send data type
            stop_replica_results, // Data array to gather
            sizeof(CrackResult),  // How many MPI_BYTES to receive per process
            MPI_BYTE,             // Receive data type
            0,                    // Root process
            MPI_COMM_WORLD        // Communicator
        );

        free(empty_crack_job);
        free(stop_jobs);
        free(stop_replica_results);
    }

    // Handle overall result
    handle_overview_result(&options, &overview_result);

    master_cleanup(shadow_file, result_file, &dict);
    return true;
}

/*
 * Initialize master stuff.
 */
static bool init_master(Options *options, Dictionary *dict, FILE **shadow_file, FILE **result_file, int argc, char **argv)
{
    int mpi_size; // Numer of processes
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Parse CLI args
    if (!parse_cli_args(options, argc, argv))
    {
        master_cleanup(*shadow_file, *result_file, dict);
        return false;
    }

    // Print some useful info
    if (!options->quiet)
    {
        printf("Workers: %d\n", mpi_size);
        printf("Max symbols: %ld\n", options->max_length);
        printf("Symbol separator: \"%s\"\n", options->separator);
    }

    // Open shadow file
    if (!options->quiet)
    {
        printf("Shadow file: %s\n", options->shadow_file);
    }
    if (!open_file(shadow_file, options->shadow_file, "r"))
    {
        master_cleanup(*shadow_file, *result_file, dict);
        return false;
    }
    // Open output file if provided
    if (options->result_file[0] != 0)
    {
        if (!options->quiet)
        {
            printf("Output file: %s\n", options->result_file);
        }
        if (!open_file(result_file, options->result_file, "w"))
        {
            master_cleanup(*shadow_file, *result_file, dict);
            return false;
        }
    }
    // Read full directory
    if (!options->quiet)
    {
        printf("Dictionary file: %s\n", options->dict_file);
    }
    if (!read_dictionary(dict, options, options->dict_file))
    {
        master_cleanup(*shadow_file, *result_file, dict);
        return false;
    }

    return true;
}

/*
 * Cleanup master stuff.
 */
static void master_cleanup(FILE *shadow_file, FILE *result_file, Dictionary *dict)
{
    if (shadow_file)
    {
        fclose(shadow_file);
    }
    if (result_file)
    {
        fclose(result_file);
    }
    if (dict->elements)
    {
        free(dict->elements);
    }
}

/*
 * Crack a shadow password entry as master.
 */
static void master_crack(ExtendedCrackResult *result, Options *options, Dictionary *dict, ShadowEntry *entry)
{

    int mpi_size; // Numer of processes
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Initialize result
    memset(result, 0, sizeof(ExtendedCrackResult));
    strncpy(result->user, entry->user, MAX_SHADOW_USER_LENGTH);
    strncpy(result->passfield, entry->passfield, MAX_SHADOW_PASSFIELD_LENGTH);
    result->alg = entry->alg;

    // Accept only known algs
    if (entry->alg == ALG_UNKNOWN)
    {
        result->status = STATUS_SKIP;
        return;
    }

    // Setup vars for cracking
    ProbeConfig config = {};
    config.dict_positions = calloc(options->max_length, sizeof(size_t));
    config.symbols = calloc(options->max_length, MAX_DICT_ELEMENT_LENGTH + 1);

    CrackJob *jobs = (CrackJob *)calloc(mpi_size, sizeof(CrackJob));                     // A memory block for all the jobs
    CrackResult *replica_results = (CrackResult *)calloc(mpi_size, sizeof(CrackResult)); // A memory block for all the results

    CrackJob *empty_crack_job = (CrackJob *)calloc(1, sizeof(CrackJob));             // Memory block for master's empty crack job
    CrackResult *empty_crack_result = (CrackResult *)calloc(1, sizeof(CrackResult)); // Memory block for master's empty crack result

    // Places the passfield into each job's passfield
    for (int i = 1; i < mpi_size; i++)
    {
        strncpy(jobs[i].passfield, entry->passfield, MAX_SHADOW_PASSFIELD_LENGTH);
    }

    // Start time measurement
    double start_time = MPI_Wtime();

    // Try probes until the status changes (when a match is found or the search space is exhausted)
    while (result->status == STATUS_PENDING)
    {
        // printf("Cracking shadow entry %s\n", entry->user);
        bool out_of_probes = false;
        // Make a job for each replica with new probes
        for (int i = 1; i < mpi_size; i++)
        {
            // memset(&jobs[i], 0, sizeof(CrackJob));
            bool more_probes = get_next_probe(&config, options, dict);
            if (!more_probes)
            {
                // * May break. Maybe set ACTION_WAIT for every remaining processs instead of continue loop
                jobs[i].action = ACTION_WAIT;
                out_of_probes = true;
                continue;
            }
            jobs[i].action = ACTION_WORK;
            strncpy(jobs[i].passfield, entry->passfield, MAX_PASSWORD_LENGTH);
            jobs[i].alg = entry->alg;
            jobs[i].salt_end = entry->salt_end;
            strncpy(jobs[i].probe, config.probe, MAX_PASSWORD_LENGTH);
            if (options->verbose)
            {
                printf("%s\n", jobs[i].probe);
            }
        }

        MPI_Scatter(
            jobs,             // The data we want to scatter
            sizeof(CrackJob), // How many MPI_BYTES to send per process -> one CrackJob
            MPI_BYTE,         // Send data type
            empty_crack_job,  // Buffer for receiving the emtpy CrackJob, i.e. only 0s
            sizeof(CrackJob), // How many MPI_BYTES to receive
            MPI_BYTE,         // Receive data type
            0,                // Root process
            MPI_COMM_WORLD    // Communicator
        );

        MPI_Gather(
            empty_crack_result,  // Data to send
            sizeof(CrackResult), // How many MPI_BYTES to send
            MPI_BYTE,            // Send data type
            replica_results,     // Data array to gather
            sizeof(CrackResult), // How many MPI_BYTES to receive per process
            MPI_BYTE,            // Receive data type
            0,                   // Root process
            MPI_COMM_WORLD       // Communicator
        );

        for (int i = 1; i < mpi_size; i++)
        {
            CrackResult *this_result = &replica_results[i];
            if (this_result->status == STATUS_SKIP)
            {
                continue;
            }
            result->attempts++;
            if (this_result->status == STATUS_SUCCESS)
            {
                result->status = STATUS_SUCCESS;
                strncpy(this_result->password, result->password, MAX_PASSWORD_LENGTH);
                break;
            }
        }

        if (out_of_probes && result->status != STATUS_SUCCESS)
        {
            result->status = STATUS_FAIL;
        }
    }

    // End time measurement
    double end_time = MPI_Wtime();
    result->duration = end_time - start_time;

    free(config.dict_positions);
    free(config.symbols);
    free(jobs);
    free(replica_results);

    free(empty_crack_job);
}

static void replica_crack(int rank, int number_of_processes)
{
    // Infinite loop: wait for a new job or ACTION_STOP
    while (true)
    {
        CrackJob *my_job = (CrackJob *)calloc(1, sizeof(CrackJob));
        CrackResult *my_result = (CrackResult *)calloc(1, sizeof(CrackResult));

        MPI_Scatter(
            NULL,             // Send buffer -> the receiver dont send anything
            0,                // Send count -> the receiver dont send anything
            MPI_BYTE,         // Send data type
            my_job,           // Receive buffer for the received CrackJob
            sizeof(CrackJob), // How many MPI_BYTES to receive -> one CrackJob
            MPI_BYTE,         // Receive data type
            0,                // Root process
            MPI_COMM_WORLD    // Communicator
        );

        if (my_job->action == ACTION_WAIT || my_job->action == ACTION_STOP)
        {
            my_result->status == STATUS_SKIP;
        }
        else
        {
            crack_job(&my_result[0], &my_job[0]);
        }

        MPI_Gather(
            my_result,           // Data to send
            sizeof(CrackResult), // How many bytes to send --> one CrackResult
            MPI_BYTE,            // Send data type
            NULL,                // Receive buffer -> nothing to receive
            0,                   // Receive count -> nothing to receive
            MPI_BYTE,            // Receive type
            0,                   // Master rank
            MPI_COMM_WORLD       // Communicator
        );

        // Stops the process
        if (my_job->action == ACTION_STOP)
        {
            break;
        }
    }
    // printf("Prrocess %d of %d finished!\n", rank, number_of_processes);
}

/*
 * Build the next probe. Returns false with an empty probe when the search space is exhausted.
 */
static bool get_next_probe(ProbeConfig *config, Options *options, Dictionary *dict)
{
    // Check if dict is empty
    if (dict->length == 0)
    {
        return false;
    }

    // Find last symbol which can be replaced with the next one, if any exists
    ssize_t last_replaceable_pos = -1;
    for (size_t i = 0; i < config->size; i++)
    {
        if (config->dict_positions[i] < dict->length - 1)
        {
            last_replaceable_pos = i;
        }
    }

    // A symbol can be replaced, replace last one and reset all behind it
    if (last_replaceable_pos >= 0)
    {
        size_t new_dict_pos = config->dict_positions[last_replaceable_pos] + 1;
        config->dict_positions[last_replaceable_pos] = new_dict_pos;
        strncpy(config->symbols[last_replaceable_pos], dict->elements[new_dict_pos], MAX_DICT_ELEMENT_LENGTH);
        for (size_t i = last_replaceable_pos + 1; i < config->size; i++)
        {
            config->dict_positions[i] = 0;
            strncpy(config->symbols[i], dict->elements[0], MAX_DICT_ELEMENT_LENGTH);
        }
    }
    // No symbols can be replaced and no more symbols are allowed, return error
    else if (config->size == options->max_length)
    {
        config->probe[0] = 0;
        return false;
    }
    // New symbol can be added, reset all previous positions and add it
    else
    {
        config->size++;
        for (size_t i = 0; i < config->size; i++)
        {
            config->dict_positions[i] = 0;
            strncpy(config->symbols[i], dict->elements[0], MAX_DICT_ELEMENT_LENGTH);
        }
    }

    // Build probe
    config->probe[0] = 0;
    for (size_t i = 0; i < config->size; i++)
    {
        if (i > 0)
        {
            strncat(config->probe, options->separator, MAX_PASSWORD_LENGTH);
        }
        strncat(config->probe, config->symbols[i], MAX_PASSWORD_LENGTH);
    }

    return true;
}

/*
 * Handle result from trying to crack a single password.
 */
static void handle_result(Options *options, ExtendedCrackResult *result, OverviewCrackResult *overview_result, FILE *result_file)
{
    // Make representations
    char const *alg_str = cryptalg_to_string(result->alg);
    char const *status_str = crack_result_status_to_string(result->status);
    double attempts_per_second = result->attempts / result->duration;

    // Format and print
    size_t const static max_output_length = 1023;
    char *output = malloc(max_output_length + 1);
    snprintf(output, max_output_length + 1, "user=\"%s\" alg=\"%s\" status=\"%s\" duration=\"%fs\" attempts=\"%ld\" attempts_per_second=\"%f\" password=\"%s\"",
             result->user, alg_str, status_str, result->duration, result->attempts, attempts_per_second, result->password);
    if (!options->quiet)
    {
        printf("%s\n", output);
    }
    if (result_file)
    {
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
static void handle_overview_result(Options *options, OverviewCrackResult *result)
{
    if (!options->quiet)
    {
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
static void crack_job(CrackResult *result, CrackJob *job)
{
    memset(result, 0, sizeof(CrackResult));

    // Only accept known (redundant check)
    if (job->alg == ALG_UNKNOWN)
    {
        result->status = STATUS_SKIP;
        return;
    }

    char const *new_passfield = crypt(job->probe, job->passfield);
    if (new_passfield != NULL && strncmp(job->passfield, new_passfield, MAX_SHADOW_PASSFIELD_LENGTH) == 0)
    {
        // Match found, abort search
        result->status = STATUS_SUCCESS;
        strncpy(result->password, job->probe, MAX_PASSWORD_LENGTH);
        result->password[MAX_PASSWORD_LENGTH] = 0;
    }
}
