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
  int image_width, image_height;
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
void applyKernel(pixel **out, pixel **in, unsigned int imageWidth, unsigned int imageHeight, int *kernel, unsigned int kernelDim, float kernelFactor)
{
  unsigned int const kernelCenter = (kernelDim / 2);
  for (unsigned int y = 0; y < imageHeight; y++)
  {
    for (unsigned int x = 0; x < imageWidth; x++)
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
          if (xx >= 0 && xx < (int)imageWidth && yy >= 0 && yy < (int)imageHeight)
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

void run_replica(int my_rank)
{

  //* Each rank gets a number of rows, not just one! (most likely)

  // TODO: Exhange borders based on my_rank % 2 == 0
  // This is so that each border pair swaps the upper/lower border,
  // which prevents deadlock, where everyone waits for e.g. the lower.

  // TODO get image dimensions from MPI_Bcast

  image_dimensions_t *image_dimensions = calloc(1, sizeof(image_dimensions_t));

  // MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
  MPI_Bcast(
      image_dimensions,           // Receive buffer
      sizeof(image_dimensions_t), // Receive one mpi_image_dimensions
      MPI_BYTE,                   // The datatype to receive
      0,                          // Master rank
      MPI_COMM_WORLD);            // Communicator

  printf("Greetings from process %d! The image is %dx%d!\n", my_rank, image_dimensions->image_width, image_dimensions->image_height);
}

int main(int argc, char **argv)
{
  MPI_Init(NULL, NULL);

  int my_rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank != 0)
  {
    run_replica(my_rank);

    MPI_Finalize();
    return 0;
  }
  else
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

    /*
    Create the BMP image and load it from disk.
   */
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

    // TODO Send image dimensions with MPI_Bcast

    image_dimensions_t *image_dimensions = calloc(1, sizeof(image_dimensions));

    image_dimensions->image_width = image->width;
    image_dimensions->image_height = image->height;

    // MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
    MPI_Bcast(
        image_dimensions,           // What to send
        sizeof(image_dimensions_t), // Send one mpi_image_dimensions
        MPI_BYTE,                   // The datatype to send
        0,                          // Master rank
        MPI_COMM_WORLD);            // Communicator

    // TODO: Use Scatterv to dispatch the rows to the processes
    // Does master get rows as well? If it does, maybe only have main function

    // Time measurement start
    double start_time = MPI_Wtime();

    // Here we do the actual computation!
    // image->data is a 2-dimensional array of pixel which is accessed row first ([y][x])
    // each pixel is a struct of 3 unsigned char for the red, blue and green colour channel
    bmpImage *processImage = newBmpImage(image->width, image->height);
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
    freeBmpImage(processImage);

    // Time measurement end
    double end_time = MPI_Wtime();

    double spentTime = end_time - start_time;
    printf("Time spent: %.3f seconds\n", spentTime);

    //Write the image back to disk
    if (saveBmpImage(image, output) != 0)
    {
      fprintf(stderr, "Could not save output to '%s'!\n", output);
      freeBmpImage(image);
      goto error_exit;
    };

  graceful_exit:
    ret = 0;
  error_exit:
    if (input)
      free(input);
    if (output)
      free(output);

    MPI_Finalize();

    return ret;
  }
};