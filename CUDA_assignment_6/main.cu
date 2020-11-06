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

#define BLOCK_DIMENSION 16 // A thread block size of 16x16 (256 threads) is a common choice (from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)

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

// Task 1-4
// Apply convolutional filter on image data
__global__ void applyFilter_CUDA_Kernel(pixel *out, pixel *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Handle out of bounds
  if (x >= width || y >= height)
  {
    return;
  }

  unsigned int const filterCenter = (filterDim / 2);
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

// Task 5
// Apply convolutional filter on image data
/*__global__ void applyFilter_CUDA_Kernel(pixel *out, pixel *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor)
{

  // Now instead of using the filter directly from global memory, we want to copy the filter to shared memory.
  // Dynamic shared memory because the filterDim is not known at compile time.

  // This one holds all of the data
  extern __shared__ int s[];

  int *shared_filter = s;                                                // Length of filterDim * filterDim
  pixel *shared_pixels = (pixel *)&shared_filter[filterDim * filterDim]; // Length of BLOCK_DIMENSION * BLOCK_DIMENSION

  for (int i = 0; i < filterDim * filterDim; i++)
  {
    shared_filter[i] = filter[i];
  }

  // Sync to make sure that all threads have completed the loads to shared memory
  __syncthreads();
  // Now we can use shared_filter!

  // Because shared memory is only shared between blocks, it makes sense to make the shared memory array for
  // the image as big as the block, since each thread in the block changes one pixel.

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Handle out of bounds
  if (x >= width || y >= height)
  {
    // __syncthreads(); // ? Needed? Think so, to avoid deadlock
    return;
  }

  // Set the position in the block to the correct value
  shared_pixels[threadIdx.y * BLOCK_DIMENSION + threadIdx.x] = in[y * width + x];

  // Sync to make sure that all threads have completed the loads to shared memory
  __syncthreads();
  // Now we can use shared_pixels!

  unsigned int const filterCenter = (filterDim / 2);
  int ar = 0, ag = 0, ab = 0;
  for (unsigned int ky = 0; ky < filterDim; ky++)
  {
    int nky = filterDim - 1 - ky;
    for (unsigned int kx = 0; kx < filterDim; kx++)
    {
      int nkx = filterDim - 1 - kx;

      int yy = y + (ky - filterCenter);
      int xx = x + (kx - filterCenter);

      // Now, since the edge threads needs pixels outside the block's shared memory,
      // we need to check its position.

      if (xx >= 0 && xx < BLOCK_DIMENSION && yy >= 0 && yy < BLOCK_DIMENSION)
      {
        ar += shared_pixels[yy * BLOCK_DIMENSION + xx].r * shared_filter[nky * filterDim + nkx];
        ag += shared_pixels[yy * BLOCK_DIMENSION + xx].g * shared_filter[nky * filterDim + nkx];
        ab += shared_pixels[yy * BLOCK_DIMENSION + xx].b * shared_filter[nky * filterDim + nkx];
      }
      // Else if the normal code from task 1-4
      else if (xx >= 0 && xx < (int)width && yy >= 0 && yy < (int)height)
      {
        ar += in[yy * width + xx].r * shared_filter[nky * filterDim + nkx];
        ag += in[yy * width + xx].g * shared_filter[nky * filterDim + nkx];
        ab += in[yy * width + xx].b * shared_filter[nky * filterDim + nkx];
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
}*/

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

  // Time measurement init
  // Inspired from https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
  cudaEvent_t start_time, end_time;
  cudaEventCreate(&start_time);
  cudaEventCreate(&end_time);

  // Here we do the actual computation!
  // image->data is a 2-dimensional array of pixel which is accessed row first ([y][x])
  // image->rawdata is a 1-dimensional array of pixel containing the same data as image->data
  // each pixel is a struct of 3 unsigned char for the red, blue and green colour channel
  // bmpImage *processImage = newBmpImage(image->width, image->height);

  int image_size = image->width * image->height * sizeof(pixel);
  int filter_size = filterDims[filterIndex] * filterDims[filterIndex] * sizeof(int);

  // We could also made all filters __device__ available, but it is simple to copy over only the needed one
  pixel *d_image_rawdata, *d_process_image_rawdata;
  int *d_filter;

  cudaMalloc((void **)&d_image_rawdata, image_size);
  cudaMalloc((void **)&d_process_image_rawdata, image_size);
  cudaMalloc((void **)&d_filter, filter_size);

  cudaMemcpy(d_image_rawdata, image->rawdata, image_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, filters[filterIndex], filter_size, cudaMemcpyHostToDevice);

  // Task 6
  // From https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
  int blockSizeInt;   // The launch configurator returned block size
  int minGridSizeInt; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
  int gridSizeInt;    // The actual grid size needed, based on input size

  cudaOccupancyMaxPotentialBlockSize(&minGridSizeInt, &blockSizeInt, applyFilter_CUDA_Kernel, 0, 0);

  // Round up according to array size
  gridSizeInt = (image->width * image->height + blockSizeInt - 1) / blockSizeInt;

  if (blockSizeInt % 32 != 0)
  {
    printf("NOTE: blockSizeInt was not a multiple of 32: %f\n", float(blockSizeInt) / 32.0);
  }

  dim3 blockSize(blockSizeInt / 32, blockSizeInt / 32);

  printf("BlockSizeInt %d\n", blockSizeInt);
  printf("minGridSizeInt %d\n", minGridSizeInt);
  printf("gridSizeInt %d\n", gridSizeInt);
  // End Task 6

  // We allocate one thread per pixel
  // gridSize and blockSize inspired from Section 2.2. in the CUDA Programming Guide
  // dim3 blockSize(BLOCK_DIMENSION, BLOCK_DIMENSION); // Threads per block
  printf("The grid has thread blocks of dimension (%d width * %d height)\n", blockSize.x, blockSize.y);

  // We may need to add 1 extra block to width or height if the image's dimensions are not evenly divided by the block's dimension
  int extraWidth = 0;
  int extraHeight = 0;

  if (image->width % blockSize.x != 0)
  {
    extraWidth = 1;
  }
  if (image->height % blockSize.y != 0)
  {
    extraHeight = 1;
  }
  dim3 gridSize(image->width / blockSize.x + extraWidth, image->height / blockSize.y + extraHeight); // Number of blocks
  printf("Launching a grid of dimension (%d width * %d height)\n", image->width / blockSize.x + extraWidth, image->height / blockSize.y + extraHeight);

  // Start time measurement
  cudaEventRecord(start_time);

  for (unsigned int i = 0; i < iterations; i++)
  {
    // Task 2-3
    applyFilter_CUDA_Kernel<<<gridSize, blockSize>>>(

        // Task 5
        // TODO Experiment with different bytes in shared memory. Share the border pixels so that we never have to access global memory for the outside bounds.
        // int sharedMemoryUsedPerBlock = filterDims[filterIndex] * filterDims[filterIndex] * sizeof(int) + BLOCK_DIMENSION * BLOCK_DIMENSION * sizeof(pixel);
        // applyFilter_CUDA_Kernel<<<gridSize, blockSize, sharedMemoryUsedPerBlock>>>(
        d_process_image_rawdata, // Out
        d_image_rawdata,         // In
        image->width,
        image->height,
        // filters[filterIndex],
        d_filter,
        filterDims[filterIndex],
        filterFactors[filterIndex]);
    // swapImage(&processImage, &image);
    swapImageRawdata(&d_process_image_rawdata, &d_image_rawdata);
  }

  // End time measurement
  cudaEventRecord(end_time);

  cudaMemcpy(image->rawdata, d_image_rawdata, image_size, cudaMemcpyDeviceToHost);

  cudaFree(d_image_rawdata);
  cudaFree(d_process_image_rawdata);
  cudaFree(d_filter);

  // Blocks CPU execution until end_time is recorded
  cudaEventSynchronize(end_time);

  float spentTime = 0.0;
  cudaEventElapsedTime(&spentTime, start_time, end_time);
  printf("Time spent: %.3f seconds\n", spentTime / 1000);

  cudaEventDestroy(start_time);
  cudaEventDestroy(end_time);

  // Task 6
  cudaDeviceSynchronize();
  // calculate theoretical occupancy
  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, applyFilter_CUDA_Kernel, blockSizeInt, 0);
  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);
  float occupancy = (maxActiveBlocks * blockSizeInt / props.warpSize) / (float)(props.maxThreadsPerMultiProcessor / props.warpSize);
  printf("Launched blocks of size %d=>(%dx%d). Theoretical occupancy: %f\n", blockSizeInt, blockSize.x, blockSize.y, occupancy);
  // End Task 6

  // Check for error
  cudaError_t error = cudaPeekAtLastError();
  if (error)
  {
    fprintf(stderr, "A CUDA error has occurred while cracking: %s\n", cudaGetErrorString(error));
  }

  //Write the image back to disk
  if (saveBmpImage(image, output) != 0)
  {
    fprintf(stderr, "Could not save output to '%s'!\n", output);
    freeBmpImage(image);
    error_exit(&input, &output);
  };

  graceful_exit(&input, &output);
};
