#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>

extern "C"
{
#include "libs/bitmap.h"
}

#define cudaErrorCheck(ans)               \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

// Convolutional Filter Examples, each with dimension 3,
// gaussian filter with dimension 5

int sobelYFilter[] = {-1, -2, -1,
                      0, 0, 0,
                      1, 2, 1};

int sobelXFilter[] = {-1, -0, 1,
                      -2, 0, 2,
                      -1, 0, 1};

int laplacian1Filter[] = {-1, -4, -1,
                          -4, 20, -4,
                          -1, -4, -1};

int laplacian2Filter[] = {0, 1, 0,
                          1, -4, 1,
                          0, 1, 0};

int laplacian3Filter[] = {-1, -1, -1,
                          -1, 8, -1,
                          -1, -1, -1};

int gaussianFilter[] = {1, 4, 6, 4, 1,
                        4, 16, 24, 16, 4,
                        6, 24, 36, 24, 6,
                        4, 16, 24, 16, 4,
                        1, 4, 6, 4, 1};

const char *filterNames[] = {"SobelY", "SobelX", "Laplacian 1", "Laplacian 2", "Laplacian 3", "Gaussian"};
int *const filters[] = {sobelYFilter, sobelXFilter, laplacian1Filter, laplacian2Filter, laplacian3Filter, gaussianFilter};
unsigned int const filterDims[] = {3, 3, 3, 3, 3, 5};
float const filterFactors[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0 / 256.0};

int const maxFilterIndex = sizeof(filterDims) / sizeof(unsigned int);

void cleanup(char **input, char **output)
{
  if (*input)
    free(*input);
  if (*output)
    free(*output);
}

void graceful_exit(char **input, char **output)
{
  cleanup(input, output);
  exit(0);
}

void error_exit(char **input, char **output)
{
  cleanup(input, output);
  exit(1);
}

// Helper function to swap bmpImageChannel pointers

void swapImageRawdata(pixel **one, pixel **two)
{
  pixel *helper = *two;
  *two = *one;
  *one = helper;
}

void swapImage(bmpImage **one, bmpImage **two)
{
  bmpImage *helper = *two;
  *two = *one;
  *one = helper;
}

// Apply convolutional filter on image data
void applyFilter(pixel *out, pixel *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor)
{
  unsigned int const filterCenter = (filterDim / 2);
  for (unsigned int y = 0; y < height; y++)
  {
    for (unsigned int x = 0; x < width; x++)
    {
      int ar = 0, ag = 0, ab = 0;
      for (unsigned int ky = 0; ky < filterDim; ky++)
      {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++)
        {
          int nkx = filterDim - 1 - kx;

          int yy = y + (ky - filterCenter);
          int xx = x + (kx - filterCenter);
          if (xx >= 0 && xx < (int)width && yy >= 0 && yy < (int)height)
          {
            ar += in[yy * width + xx].r * filter[nky * filterDim + nkx];
            ag += in[yy * width + xx].g * filter[nky * filterDim + nkx];
            ab += in[yy * width + xx].b * filter[nky * filterDim + nkx];
          }
        }
      }

      ar *= filterFactor;
      ag *= filterFactor;
      ab *= filterFactor;

      ar = (ar < 0) ? 0 : ar;
      ag = (ag < 0) ? 0 : ag;
      ab = (ab < 0) ? 0 : ab;

      out[y * width + x].r = (ar > 255) ? 255 : ar;
      out[y * width + x].g = (ag > 255) ? 255 : ag;
      out[y * width + x].b = (ab > 255) ? 255 : ab;
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
  fprintf(out, "  -k, --filter     <filter>        filter index (0<=x<=%u) (2)\n", maxFilterIndex - 1);
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
  unsigned int filterIndex = 2;

  static struct option const long_options[] = {
      {"help", no_argument, 0, 'h'},
      {"filter", required_argument, 0, 'k'},
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
        graceful_exit(&input, &output);
      case 'k':
        parse = strtol(optarg, &endptr, 10);
        if (endptr == optarg || parse < 0 || parse >= maxFilterIndex)
        {
          help(argv[0], c, optarg);
          error_exit(&input, &output);
        }
        filterIndex = (unsigned int)parse;
        break;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg)
        {
          help(argv[0], c, optarg);
          error_exit(&input, &output);
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
    error_exit(&input, &output);
  }

  unsigned int arglen = strlen(argv[optind]);
  input = (char *)calloc(arglen + 1, sizeof(char));
  strncpy(input, argv[optind], arglen);
  optind++;

  arglen = strlen(argv[optind]);
  output = (char *)calloc(arglen + 1, sizeof(char));
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
    error_exit(&input, &output);
  }

  if (loadBmpImage(image, input) != 0)
  {
    fprintf(stderr, "Could not load bmp image '%s'!\n", input);
    freeBmpImage(image);
    error_exit(&input, &output);
  }

  printf("Apply filter '%s' on image with %u x %u pixels for %u iterations\n", filterNames[filterIndex], image->width, image->height, iterations);

  // TODO: implement time measurement from here

  // Here we do the actual computation!
  // image->data is a 2-dimensional array of pixel which is accessed row first ([y][x])
  // image->rawdata is a 1-dimensional array of pixel containing the same data as image->data
  // each pixel is a struct of 3 unsigned char for the red, blue and green colour channel
  bmpImage *processImage = newBmpImage(image->width, image->height);

  // TODO: Cuda malloc and memcpy the rawdata from the images, from host side to device side

  // TODO: Define the gridSize and blockSize, e.g. using dim3 (see Section 2.2. in CUDA Programming Guide)

  // TODO: Intialize and start CUDA timer

  clock_t start_time, end_time;
  double cpu_time_used;

  start_time = clock();
  for (unsigned int i = 0; i < iterations; i++)
  {
    // TODO: Implement kernel call instead of serial implementation
    applyFilter(processImage->rawdata,
                image->rawdata,
                image->width,
                image->height,
                filters[filterIndex],
                filterDims[filterIndex],
                filterFactors[filterIndex]);
    swapImage(&processImage, &image);
    //swapImageRawdata(&?, &?);
  }

  end_time = clock();
  cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

  // TODO: Stop CUDA timer

  // TODO: Copy back rawdata from images

  // TODO: Calculate and print elapsed time
  float spentTime = 0.0;
  printf("Time spent: %.3f seconds\n", cpu_time_used);

  freeBmpImage(processImage);
  //Write the image back to disk
  if (saveBmpImage(image, output) != 0)
  {
    fprintf(stderr, "Could not save output to '%s'!\n", output);
    freeBmpImage(image);
    error_exit(&input, &output);
  };

  graceful_exit(&input, &output);
};
