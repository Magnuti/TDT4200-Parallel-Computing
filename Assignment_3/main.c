#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include "libs/bitmap.h"

#include <mpi.h>

typedef struct
{
  unsigned int width, height, kernelIndex;
} image_dimensions_t;

// Convolutional Kernel Examples, each with dimension 3,
// gaussian kernel with dimension 5

int sobelYKernel[] = {-1, -2, -1,
                      0, 0, 0,
                      1, 2, 1};

int sobelXKernel[] = {-1, -0, 1,
                      -2, 0, 2,
                      -1, 0, 1};

int laplacian1Kernel[] = {-1, -4, -1,
                          -4, 20, -4,
                          -1, -4, -1};

int laplacian2Kernel[] = {0, 1, 0,
                          1, -4, 1,
                          0, 1, 0};

int laplacian3Kernel[] = {-1, -1, -1,
                          -1, 8, -1,
                          -1, -1, -1};

int gaussianKernel[] = {1, 4, 6, 4, 1,
                        4, 16, 24, 16, 4,
                        6, 24, 36, 24, 6,
                        4, 16, 24, 16, 4,
                        1, 4, 6, 4, 1};

char *const kernelNames[] = {"SobelY", "SobelX", "Laplacian 1", "Laplacian 2", "Laplacian 3", "Gaussian"};
int *const kernels[] = {sobelYKernel, sobelXKernel, laplacian1Kernel, laplacian2Kernel, laplacian3Kernel, gaussianKernel};
unsigned int const kernelDims[] = {3, 3, 3, 3, 3, 5};
float const kernelFactors[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0 / 256.0};

int const maxKernelIndex = sizeof(kernelDims) / sizeof(unsigned int);

// Helper function to swap bmpImageChannel pointers

void swapImage(bmpImage **one, bmpImage **two)
{
  bmpImage *helper = *two;
  *two = *one;
  *one = helper;
}

// Apply convolutional kernel on image data
void applyKernel(pixel **out, pixel **in, unsigned int width, unsigned int height, int *kernel, unsigned int kernelDim, float kernelFactor)
{
  unsigned int const kernelCenter = (kernelDim / 2);
  for (unsigned int y = 0; y < height; y++)
  {
    for (unsigned int x = 0; x < width; x++)
    {
      unsigned int ar = 0, ag = 0, ab = 0;
      for (unsigned int ky = 0; ky < kernelDim; ky++)
      {
        int nky = kernelDim - 1 - ky;
        for (unsigned int kx = 0; kx < kernelDim; kx++)
        {
          int nkx = kernelDim - 1 - kx;

          int yy = y + (ky - kernelCenter);
          int xx = x + (kx - kernelCenter);
          if (xx >= 0 && xx < (int)width && yy >= 0 && yy < (int)height)
          {
            ar += in[yy][xx].r * kernel[nky * kernelDim + nkx];
            ag += in[yy][xx].g * kernel[nky * kernelDim + nkx];
            ab += in[yy][xx].b * kernel[nky * kernelDim + nkx];
          }
        }
      }
      if (ar || ag || ab)
      {
        ar *= kernelFactor;
        ag *= kernelFactor;
        ab *= kernelFactor;
        out[y][x].r = (ar > 255) ? 255 : ar;
        out[y][x].g = (ag > 255) ? 255 : ag;
        out[y][x].b = (ab > 255) ? 255 : ab;
      }
      else
      {
        out[y][x].r = 0;
        out[y][x].g = 0;
        out[y][x].b = 0;
      }
    }
  }
}

void help(char const *exec, char const opt, char const *optarg)
{
  FILE *out = stdout;
  if (opt != 0)
  {
    out = stderr;
    if (optarg)
    {
      fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
    }
    else
    {
      fprintf(out, "Invalid parameter - %c\n", opt);
    }
  }
  fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
  fprintf(out, "\n");
  fprintf(out, "Options:\n");
  fprintf(out, "  -k, --kernel     <kernel>        kernel index (0<=x<=%u) (2)\n", maxKernelIndex - 1);
  fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

  fprintf(out, "\n");
  fprintf(out, "Example: %s before.bmp after.bmp -i 10000\n", exec);
}

int main(int argc, char **argv)
{
  /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  unsigned int kernelIndex = 2;
  int ret = 0;

  static struct option const long_options[] = {
      {"help", no_argument, 0, 'h'},
      {"kernel", required_argument, 0, 'k'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}};

  static char const *short_options = "hk:i:";
  {
    char *endptr;
    int c;
    int parse;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1)
    {
      switch (c)
      {
      case 'h':
        help(argv[0], 0, NULL);
        goto graceful_exit;
      case 'k':
        parse = strtol(optarg, &endptr, 10);
        if (endptr == optarg || parse < 0 || parse >= maxKernelIndex)
        {
          help(argv[0], c, optarg);
          goto error_exit;
        }
        kernelIndex = (unsigned int)parse;
        break;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg)
        {
          help(argv[0], c, optarg);
          goto error_exit;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind + 1))
  {
    help(argv[0], ' ', "Not enough arugments");
    goto error_exit;
  }

  unsigned int arglen = strlen(argv[optind]);
  input = calloc(arglen + 1, sizeof(char));
  strncpy(input, argv[optind], arglen);
  optind++;

  arglen = strlen(argv[optind]);
  output = calloc(arglen + 1, sizeof(char));
  strncpy(output, argv[optind], arglen);
  optind++;

  /*
    End of Parameter parsing!
   */

  MPI_Init(NULL, NULL);
  //* ? MPI_Init(&argc, &argv);

  int my_rank, world_size;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  image_dimensions_t *imageDimensions = calloc(1, sizeof(image_dimensions_t));
  pixel *sendBuffer;
  pixel *receiveBuffer;

  double startTime, endTime;

  /*
    Create the BMP image and load it from disk.
   */
  if (my_rank == 0)
  {
    bmpImage *image = newBmpImage(0, 0);
    if (image == NULL)
    {
      fprintf(stderr, "Could not allocate new image!\n");
      goto error_exit;
    }

    if (loadBmpImage(image, input) != 0)
    {
      fprintf(stderr, "Could not load bmp image '%s'!\n", input);
      freeBmpImage(image);
      goto error_exit;
    }
    printf("Apply kernel '%s' on image with %u x %u pixels for %u iterations\n", kernelNames[kernelIndex], image->width, image->height, iterations);

    imageDimensions->width = image->width;
    imageDimensions->height = image->height;
    imageDimensions->kernelIndex = kernelIndex;

    // Copy image->rawdata
    sendBuffer = (pixel *)calloc(image->width * image->height, sizeof(pixel));
    for (int i = 0; i < image->height * image->width; i++)
    {
      sendBuffer[i] = image->rawdata[i]; //* Hmm
    }

    // Same size as sendBuffer of course
    receiveBuffer = (pixel *)calloc(image->width * image->height, sizeof(pixel));

    freeBmpImage(image);

    // Start time measurement before any MPI communication takes place
    startTime = MPI_Wtime();
  }

  // MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
  MPI_Bcast(
      imageDimensions,            // Send buffer for master, receive buffer for replicas
      sizeof(image_dimensions_t), // Send/receive one image_dimensions_t
      MPI_BYTE,                   // The datatype to send
      0,                          // Master rank
      MPI_COMM_WORLD);            // Communicator

  printf("Greetings from process %d! The image is %dx%d!\n", my_rank, imageDimensions->width, imageDimensions->height);

  /* Start row scatter */
  {
    int rowSize = imageDimensions->width * sizeof(pixel); // Width * sizeof(pixel)
    int send_counts[world_size];                          // How many bytes (rows * row_size) to send
    int displacements[world_size];                        // Offset for where the data is located for each process on the array
    int min_rows_per_process = imageDimensions->height / world_size;
    int remainder_rows = imageDimensions->height % world_size;

    // Divide the rows as evenly as possible among processes, the first ones may get one extra row. E.g. [2, 2, 1]
    for (int i = 0; i < world_size; i++)
    {
      int rows = i < remainder_rows ? min_rows_per_process + 1 : min_rows_per_process;
      send_counts[i] = rows * rowSize;
    }

    displacements[0] = 0;
    for (int i = 1; i < world_size; i++)
    {
      displacements[i] = displacements[i - 1] + send_counts[i - 1];
      if (my_rank == 0)
      {
        // printf("Displacements %d: %d:\n", i, displacements[i]);
        // Maybe entire displacements should go into master
      }
    }

    pixel *my_rows = calloc(1, send_counts[my_rank]);

    // MPI_Scatterv(send_buffer, send_counts, displacements, send_type, recv_buffer, recv_count, recv_type, 0, communicator);
    MPI_Scatterv(
        sendBuffer,           // Send buffer
        send_counts,          // Array of length world_size -> how many rows to send to each process e.g. [2, 2, 1] * row_size
        displacements,        // Array of length world_size -> the offset of the send_buffer of where to start sending data to a process
        MPI_BYTE,             // Send type
        my_rows,              // Receive buffer
        send_counts[my_rank], // Receive count
        MPI_BYTE,             // Receive type
        0,
        MPI_COMM_WORLD);

    int number_of_my_rows = send_counts[my_rank] / rowSize;

    // printf("Process %d received %d rows\n", my_rank, number_of_my_rows);

    /* End scatter */

    /* Start border swapping and image processing  */

    int number_of_rows_to_swap = kernelDims[kernelIndex] / 2;

    bmpImage *my_image_rows_and_borders = newBmpImage(imageDimensions->width, number_of_my_rows + number_of_rows_to_swap * 2);

    // Puts my_rows inside my_image_rows_and_borders
    //* Inefficient code..
    for (int y = 0; y < number_of_my_rows; y++)
    {
      for (int x = 0; x < imageDimensions->width; x++)
      {
        my_image_rows_and_borders->data[number_of_rows_to_swap + y][x] = my_rows[y * imageDimensions->width + x];
      }
    }

    free(my_rows);

    int bytesToSwap = imageDimensions->width * number_of_rows_to_swap * sizeof(pixel);

    for (unsigned int i = 0; i < iterations; i++)
    {

      // Ignore border exhange if we only have one process
      if (world_size > 1)
      {
        // TODO Swap row multiple times for matrix > 3x3 -> OR, find a way to send it all at once.
        // Even ranks swap right first, then left
        if (my_rank % 2 == 0)
        {
          // * Maybe make these into functions instead of repeating them
          if (my_rank < world_size - 1)
          {
            // printf("Rank %d tries to send/receive %d bytes with rank %d The offset is %ld.\n", my_rank, bytesToSwap, my_rank + 1, send_counts[my_rank] / sizeof(pixel) - pixels_to_swap);
            MPI_Sendrecv(
                // &my_rows[(send_counts[my_rank] / sizeof(pixel)) - pixels_to_swap], // Send the last rows in my_rows
                my_image_rows_and_borders->data[my_image_rows_and_borders->height - 2], // Send buffer lower rows // ! Fix -1
                bytesToSwap,                                                            // How many MPI_BYTES to send
                MPI_BYTE,                                                               // Send type
                my_rank + 1,                                                            // Receiver rank
                0,                                                                      // Tag
                my_image_rows_and_borders->data[my_image_rows_and_borders->height - 1], // Receive buffer lower rows
                bytesToSwap,                                                            // Receive count in MPI_BYTES
                MPI_BYTE,                                                               // Receive data type
                my_rank + 1,                                                            // Source rank
                0,                                                                      // Tag
                MPI_COMM_WORLD,                                                         // Communicator
                MPI_STATUS_IGNORE                                                       // Status
            );
          }
          if (my_rank > 0)
          {
            // printf("Rank %d tries to send/receive %d bytes with rank %d.\n", my_rank, bytesToSwap, my_rank - 1);
            MPI_Sendrecv(
                my_image_rows_and_borders->data[1], // Send buffer upper rows // ! Fix +1
                bytesToSwap,                        // How many MPI_BYTES to send
                MPI_BYTE,                           // Send type
                my_rank - 1,                        // Receiver rank
                0,                                  // Tag
                my_image_rows_and_borders->data[0], // Receive buffer upper rows
                bytesToSwap,                        // Receive count in MPI_BYTES
                MPI_BYTE,                           // Receive data type
                my_rank - 1,                        // Source rank
                0,                                  // Tag
                MPI_COMM_WORLD,                     // Communicator
                MPI_STATUS_IGNORE                   // Status
            );
          }
        }
        // Odd ranks swap left first, then right
        else
        {
          // printf("Rank %d tries to send/receive %d bytes with rank %d The offset is %ld.\n", my_rank, bytesToSwap, my_rank - 1, send_counts[my_rank] / sizeof(pixel) - pixels_to_swap);
          MPI_Sendrecv(
              my_image_rows_and_borders->data[1], // Send buffer upper rows // ! Fix +1
              bytesToSwap,                        // How many MPI_BYTES to send
              MPI_BYTE,                           // Send type
              my_rank - 1,                        // Receiver rank
              0,                                  // Tag
              my_image_rows_and_borders->data[0], // Receive buffer upper rows
              bytesToSwap,                        // Receive count in MPI_BYTES
              MPI_BYTE,                           // Receive data type
              my_rank - 1,                        // Source rank
              0,                                  // Tag
              MPI_COMM_WORLD,                     // Communicator
              MPI_STATUS_IGNORE                   // Status
          );
          // printf("Rank %d tries to send/receive %d bytes with rank %d.\n", my_rank, bytesToSwap, my_rank + 1);
          if (my_rank < world_size - 1)
          {
            MPI_Sendrecv(
                // &my_rows[(send_counts[my_rank] / sizeof(pixel)) - pixels_to_swap], // Send the last rows in my_rows
                my_image_rows_and_borders->data[my_image_rows_and_borders->height - 2], // Send buffer lower rows // ! Fix
                bytesToSwap,                                                            // How many MPI_BYTES to send
                MPI_BYTE,                                                               // Send type
                my_rank + 1,                                                            // Receiver rank
                0,                                                                      // Tag
                my_image_rows_and_borders->data[my_image_rows_and_borders->height - 1], // Receive buffer lower rows
                bytesToSwap,                                                            // Receive count in MPI_BYTES
                MPI_BYTE,                                                               // Receive data type
                my_rank + 1,                                                            // Source rank
                0,                                                                      // Tag
                MPI_COMM_WORLD,                                                         // Communicator
                MPI_STATUS_IGNORE                                                       // Status
            );
          }
        }
      }

      printf("Rank %d completed border exhange for iteration %d\n", my_rank, 1);

      // Now we have the borders, so we can start the convolution with our kernel

      bmpImage *processImage = newBmpImage(my_image_rows_and_borders->width, my_image_rows_and_borders->height);
      applyKernel(processImage->data,
                  my_image_rows_and_borders->data,
                  my_image_rows_and_borders->width,
                  my_image_rows_and_borders->height,
                  kernels[kernelIndex],
                  kernelDims[kernelIndex],
                  kernelFactors[kernelIndex]);
      swapImage(&processImage, &my_image_rows_and_borders);
      freeBmpImage(processImage);
    }

    /* End border swapping and image processing  */

    /* Start the gathering! The image is now fully processed, only combining it remains. */

    // TODO: use my_image_rows_and_borders?
    // bmpImage *my_image_rows_and_borders = newBmpImage(image_dimensions->image_width, number_of_my_rows + number_of_rows_to_swap * 2);

    // TODO: Use ->rawData instead
    // Use a 1D array so that it is easier to gather
    pixel *flat_image = calloc(1, send_counts[my_rank]);
    for (int y = 0; y < number_of_my_rows; y++)
    {
      for (int x = 0; x < imageDimensions->width; x++)
      {
        flat_image[y * imageDimensions->width + x] = my_image_rows_and_borders->data[y + number_of_rows_to_swap][x];
      }
    }

    //MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,void *recvbuf, const int *recvcounts, const int *displs,MPI_Datatype recvtype, int root, MPI_Comm comm);
    MPI_Gatherv(
        flat_image,           // Send buffer
        send_counts[my_rank], // Send the entire buffer
        MPI_BYTE,             // Send type
        receiveBuffer,        // Receive buffer
        send_counts,          //
        displacements,        //
        MPI_BYTE,             // Receive type
        0,                    // Master rank
        MPI_COMM_WORLD);      // Communicator

    /* End gather */
  }

  // Here we do the actual computation!
  // image->data is a 2-dimensional array of pixel which is accessed row first ([y][x])
  // each pixel is a struct of 3 unsigned char for the red, blue and green colour channel
  /*bmpImage *processImage = newBmpImage(image->width, image->height);
  for (unsigned int i = 0; i < iterations; i++)
  {
    applyKernel(processImage->data,
                image->data,
                image->width,
                image->height,
                kernels[kernelIndex],
                kernelDims[kernelIndex],
                kernelFactors[kernelIndex]);
    swapImage(&processImage, &image);
  }
  freeBmpImage(processImage);*/

  if (my_rank == 0)
  {
    double endTime = MPI_Wtime();
    printf("Time spent: %.3f seconds\n", endTime - startTime);
  }

  // Save the image
  if (my_rank == 0)
  {
    bmpImage *saveImage = newBmpImage(imageDimensions->width, imageDimensions->height);

    for (int i = 0; i < imageDimensions->height * imageDimensions->width; i++)
    {
      saveImage->rawdata[i] = receiveBuffer[i];
    }

    //Write the image back to disk
    if (saveBmpImage(saveImage, output) != 0)
    {
      fprintf(stderr, "Could not save output to '%s'!\n", output);
      freeBmpImage(saveImage);
      goto error_exit;
    };
  }

graceful_exit:
  ret = 0;
error_exit:
  if (input)
    free(input);
  if (output)
    free(output);

  MPI_Finalize();

  return ret;
};
