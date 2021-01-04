#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_fp16.h>
#include <mma.h> // CUDA WMMA API

using namespace nvcuda;

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

// Define some error checking macros.
#define cudaErrCheck(stat)                     \
  {                                            \
    cudaErrCheck_((stat), __FILE__, __LINE__); \
  }
void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
  if (stat != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
  }
}

#define WARP_SIZE 32
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
// float const filterFactors[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0 / 256.0}; // Not used in this case

const unsigned int MAX_GRID_DIMENSION = 65535;

// const unsigned int numberOfChannels = 3;

// Hardcoded selected filters for now
const unsigned int numberOfFiltersUsed = 5;
const unsigned int filterIndexes[] = {0, 1, 2, 3, 4};
const unsigned int filterDim = 3; // Only one filter dimension supported for now

// WMMA stuff from https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// mxk * kxn = mxn
const unsigned int DESIRED_M = numberOfFiltersUsed; // 5 filters
const unsigned int DESIRED_K = 9;                   // 3x3 filters
const unsigned int DESIRED_N = 4000 * 2334;         // 4000x2334 image

// Must be multiplies of 16
const int MATRIX_M = 16;        // I want 5 here (5 filters)
const int MATRIX_K = 16;        // I want 9 here (3x3 filter values)
const int MATRIX_N = DESIRED_N; // It is evenly divisible by 16

// im2col and col2im taken from https://github.com/pluskid/Mocha.jl/blob/master/deps/im2col.cpp#L7
// The function works on one channel at the time because the output image should not be only one channel.
// Furthemore, we assume that we use a valid padding (input dim = output dim) and 1 stride.
void im2col(pixel *img, unsigned char *col, int width, int height, int filterDim, int color)
{
  // The dimension of the col array is im_width*im_height*filter_h*filter_w
  int kernel_h = filterDim,  // Assuming square kernel
      kernel_w = filterDim,  // Assuming square kernel
      pad_h = filterDim / 2, // Same padding
      pad_w = filterDim / 2; // Same padding
  int height_col = (height + 2 * pad_h - kernel_h) + 1;
  int width_col = (width + 2 * pad_w - kernel_w) + 1;
  int channels_col = kernel_h * kernel_w;

  for (int c = 0; c < channels_col; ++c)
  {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / (kernel_h * kernel_w);

    for (int h = 0; h < height_col; ++h)
    {
      for (int w = 0; w < width_col; ++w)
      {
        int h_pad = h - pad_h + h_offset;
        int w_pad = w - pad_w + w_offset;
        int index_col = (c * height_col + h) * width_col + w;
        int index_im = (c_im * height + h_pad) * width + w_pad;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
        {
          if (color == 0)
          {
            col[index_col] = img[index_im].r;
          }
          else if (color == 1)
          {
            col[index_col] = img[index_im].g;
          }
          else
          {
            col[index_col] = img[index_im].b;
          }
        }
        else
        {
          col[index_col] = 0;
        }
      }
    }
  }
}

void buildFilterArray(float *array)
{
  // The filter is of dimension mxk
  // The 0s padding is included in this function
  for (int i = 0; i < MATRIX_M; i++)
  {
    for (int j = 0; j < MATRIX_K; j++)
    {
      int arrayIndex = i * MATRIX_K + j;
      if (i < DESIRED_M && j < DESIRED_K)
      {
        unsigned int filterIndex = filterIndexes[i];
        array[arrayIndex] = (float)filters[filterIndex][j];
      }
      else
      {
        array[arrayIndex] = 0.0f;
      }
    }
  }
}

void buildImageArray(unsigned char *outputCol, unsigned char *inputCol, unsigned int intputColLength)
{
  // The image is of dimension kxn
  // This function adds the 5 0s to the colums
  memcpy(outputCol, inputCol, intputColLength * sizeof(unsigned char));
  for (int k = 0; k < MATRIX_K - DESIRED_K; k++)
  {
    for (int n = 0; n < MATRIX_N; n++)
    {
      outputCol[intputColLength + k * MATRIX_N + n] = 0;
    }
  }
}

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

// This function is only used for creating the baseline images for filter 0-4
// Apply convolutional filter on image data with the use of shared memory
__global__ void apply_filter(pixel *out, pixel *in, unsigned int width, unsigned int height, int *filter, unsigned int filterDim, float filterFactor)
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
}

// __global__ void apply_filter_GEMM(bmpImage *out, bmpImage *in, int *filters, int numberOfFilters, unsigned int filterDim, unsigned int filterSize, float filterFactor)
__global__ void apply_filter_GEMM(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta)
{
  // Leading dimensions. Packed with no transpositions.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // From https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-example
  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag; // This holds A*B
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  // Initialize the output to zero
  wmma::fill_fragment(acc_frag, 0.0f);

  // A*B
  // Loop over the K-dimension
  for (int i = 0; i < K; i += WMMA_K)
  {
    int aRow = warpM * WMMA_M;
    int aCol = i;
    int bRow = i;
    int bCol = warpN * WMMA_N;

    // Bounds checking
    if (aRow < M && aCol < K && bRow < K && bCol < N)
    {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // AB + C
  // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;

  if (cRow < M && cCol < N)
  {
    wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

    for (int i = 0; i < c_frag.num_elements; i++)
    {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
  }
}

__host__ void floatToHalf(half *out, float *in, int n)
{
  for (int i = 0; i < n; i++)
  {
    out[i] = __float2half(in[i]);
  }
}

// __global__ void convertFp32ToFp16(half *out, float *in, int n)
// {
//   int idx = blockDim.x * blockIdx.x + threadIdx.x;
//   if (idx < n)
//   {
//     out[idx] = in[idx];
//   }
// }

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

  static struct option const long_options[] = {
      {"help", no_argument, 0, 'h'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}};

  static char const *short_options = "h:i:";
  {
    char *endptr;
    int c;
    // int parse;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1)
    {
      switch (c)
      {
      case 'h':
        help(argv[0], 0, NULL);
        graceful_exit(&input, &output);
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

  // Const initialization
  for (unsigned int i = 1; i < numberOfFiltersUsed; i++)
  {
    if (filterDims[filterIndexes[i]] != filterDim)
    {
      printf("Unequal filter dimension used");
      exit(1);
    }
  }

  float *filterCol = (float *)malloc(MATRIX_M * MATRIX_K * sizeof(float));
  buildFilterArray(filterCol);

  unsigned int tempImageColLength = image->width * image->height * filterDim * filterDim;
  unsigned char *tempImageCol_r = (unsigned char *)malloc(tempImageColLength * sizeof(unsigned char));
  unsigned char *tempImageCol_g = (unsigned char *)malloc(tempImageColLength * sizeof(unsigned char));
  unsigned char *tempImageCol_b = (unsigned char *)malloc(tempImageColLength * sizeof(unsigned char));
  im2col(image->rawdata, tempImageCol_r, image->width, image->height, filterDim, 0);
  im2col(image->rawdata, tempImageCol_g, image->width, image->height, filterDim, 1);
  im2col(image->rawdata, tempImageCol_b, image->width, image->height, filterDim, 2);

  unsigned char *imageColChar_r = (unsigned char *)malloc(MATRIX_K * MATRIX_N * sizeof(unsigned char));
  unsigned char *imageColChar_g = (unsigned char *)malloc(MATRIX_K * MATRIX_N * sizeof(unsigned char));
  unsigned char *imageColChar_b = (unsigned char *)malloc(MATRIX_K * MATRIX_N * sizeof(unsigned char));
  buildImageArray(imageColChar_r, tempImageCol_r, tempImageColLength);
  buildImageArray(imageColChar_g, tempImageCol_g, tempImageColLength);
  buildImageArray(imageColChar_b, tempImageCol_b, tempImageColLength);

  // TOOD try to fix these, it crashes on free _b
  tempImageCol_r = NULL;
  tempImageCol_g = NULL;
  tempImageCol_b = NULL;
  free(tempImageCol_r);
  free(tempImageCol_g);
  free(tempImageCol_b);

  float *imageCol_r = (float *)malloc(MATRIX_K * MATRIX_N * sizeof(float));
  float *imageCol_g = (float *)malloc(MATRIX_K * MATRIX_N * sizeof(float));
  float *imageCol_b = (float *)malloc(MATRIX_K * MATRIX_N * sizeof(float));
  for (int i = 0; i < MATRIX_K * MATRIX_N; i++)
  {
    imageCol_r[i] = (float)imageColChar_r[i];
    imageCol_g[i] = (float)imageColChar_g[i];
    imageCol_b[i] = (float)imageColChar_b[i];
  }

  imageColChar_r = NULL;
  imageColChar_g = NULL;
  imageColChar_b = NULL;
  free(imageColChar_r);
  free(imageColChar_g);
  free(imageColChar_b);

  printf("Apply filters ");
  for (size_t i = 0; i < sizeof(filterIndexes) / sizeof(filterIndexes[0]); i++)
  {
    printf("%s ", filterNames[i]);
  }
  printf("on image with %u x %u pixels for %u iterations\n", image->width, image->height, iterations);

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

  half *a_fp16_host = (half *)malloc(MATRIX_M * MATRIX_K * sizeof(half));
  half *b_fp16_r_host = (half *)malloc(MATRIX_K * MATRIX_N * sizeof(half));
  half *b_fp16_g_host = (half *)malloc(MATRIX_K * MATRIX_N * sizeof(half));
  half *b_fp16_b_host = (half *)malloc(MATRIX_K * MATRIX_N * sizeof(half));

  printf("To half?\n");
  // Convert float to halves, could also be done more efficiently on the GPU, but this is a simple solution.
  floatToHalf(a_fp16_host, filterCol, MATRIX_M * MATRIX_K);
  floatToHalf(b_fp16_r_host, imageCol_r, MATRIX_K * MATRIX_N);
  floatToHalf(b_fp16_g_host, imageCol_g, MATRIX_K * MATRIX_N);
  floatToHalf(b_fp16_b_host, imageCol_b, MATRIX_K * MATRIX_N);
  printf("To half!\n");

  // All taken from https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu
  // float *a_fp32;        // Filter temp
  // float *b_fp32_r;      // Image temp
  // float *b_fp32_g;      // Image temp
  // float *b_fp32_b;      // Image temp
  half *a_fp16;         // Filter
  half *b_fp16_r;       // Image array
  half *b_fp16_g;       // Image array
  half *b_fp16_b;       // Image array
  float *c_wmma_r;      // Device answer array
  float *c_wmma_g;      // Device answer array
  float *c_wmma_b;      // Device answer array
  float *c_host_wmma_r; // Host answer array
  float *c_host_wmma_g; // Host answer array
  float *c_host_wmma_b; // Host answer array

  // cudaErrCheck(cudaMalloc((void **)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));

  // cudaErrCheck(cudaMalloc((void **)&b_fp32_r, MATRIX_K * MATRIX_N * sizeof(float)));
  // cudaErrCheck(cudaMalloc((void **)&b_fp32_g, MATRIX_K * MATRIX_N * sizeof(float)));
  // cudaErrCheck(cudaMalloc((void **)&b_fp32_b, MATRIX_K * MATRIX_N * sizeof(float)));

  cudaErrCheck(cudaMalloc((void **)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));

  cudaErrCheck(cudaMalloc((void **)&b_fp16_r, MATRIX_K * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc((void **)&b_fp16_g, MATRIX_K * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc((void **)&b_fp16_b, MATRIX_K * MATRIX_N * sizeof(half)));

  cudaErrCheck(cudaMalloc((void **)&c_wmma_r, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void **)&c_wmma_g, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void **)&c_wmma_b, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMemset(c_wmma_r, 0.0f, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMemset(c_wmma_g, 0.0f, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMemset(c_wmma_b, 0.0f, MATRIX_M * MATRIX_N * sizeof(float)));

  c_host_wmma_r = (float *)calloc(sizeof(float), MATRIX_M * MATRIX_N);
  c_host_wmma_g = (float *)calloc(sizeof(float), MATRIX_M * MATRIX_N);
  c_host_wmma_b = (float *)calloc(sizeof(float), MATRIX_M * MATRIX_N);

  printf("Copying over halves?\n");
  cudaErrCheck(cudaMemcpy(a_fp16, a_fp16_host, MATRIX_M * MATRIX_K * sizeof(half), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(b_fp16_r, b_fp16_r_host, MATRIX_K * MATRIX_N * sizeof(half), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(b_fp16_g, b_fp16_g_host, MATRIX_K * MATRIX_N * sizeof(half), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(b_fp16_b, b_fp16_b_host, MATRIX_K * MATRIX_N * sizeof(half), cudaMemcpyHostToDevice));
  printf("Copying over halves!\n");

  // cudaErrCheck(cudaMemcpy(b_fp32_r, imageCol_r, MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
  // cudaErrCheck(cudaMemcpy(b_fp32_g, imageCol_g, MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
  // cudaErrCheck(cudaMemcpy(b_fp32_b, imageCol_b, MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));

  // cudaErrCheck(cudaDeviceSynchronize());
  filterCol = NULL;
  imageCol_r = NULL;
  imageCol_g = NULL;
  imageCol_b = NULL;
  free(filterCol);
  free(imageCol_r);
  free(imageCol_g);
  free(imageCol_b);

  // curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
  // curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
  // Convert float to half
  // int filterGridDim = (MATRIX_M * MATRIX_K + 255) / 256;
  // int filterBlockDim = 256;
  // printf("Float to half kernel launch with grid dim %d, block dim %d\n", filterGridDim, filterBlockDim);
  // // convertFp32ToFp16<<<filterGridDim, filterBlockDim>>>(a_fp16, a_fp32, MATRIX_M * MATRIX_K);
  // toHalf(a_fp16, a_fp32, MATRIX_M * MATRIX_K);
  // int imageGridDim = (MATRIX_K * MATRIX_N + 255) / 256;
  // int imageBlockDim = 256;
  // printf("Float to half kernel launch with grid dim %d, block dim %d\n", imageGridDim, imageBlockDim);
  // toHalf(b_fp16_r, b_fp32_r, MATRIX_K * MATRIX_N);
  // toHalf(b_fp16_g, b_fp32_g, MATRIX_K * MATRIX_N);
  // toHalf(b_fp16_b, b_fp32_b, MATRIX_K * MATRIX_N);
  // convertFp32ToFp16<<<imageGridDim, imageBlockDim>>>(b_fp16_r, b_fp32_r, MATRIX_K * MATRIX_N);
  // convertFp32ToFp16<<<imageGridDim, imageBlockDim>>>(b_fp16_g, b_fp32_g, MATRIX_K * MATRIX_N);
  // convertFp32ToFp16<<<imageGridDim, imageBlockDim>>>(b_fp16_b, b_fp32_b, MATRIX_K * MATRIX_N);

  // printf("Before c_wmma copy\n");
  // cudaErrCheck(cudaMemcpy(c_wmma_r, c_host_wmma_r, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
  // cudaErrCheck(cudaMemcpy(c_wmma_g, c_host_wmma_g, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
  // cudaErrCheck(cudaMemcpy(c_wmma_b, c_host_wmma_b, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
  // printf("After c_wmma copy\n");

  // curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));

  // curandErrCheck(curandDestroyGenerator(gen));

  dim3 gridDim;
  dim3 blockDim;

  // blockDim.x must be a multple of warpSize
  // 16 warps in one block and a block computes a 64x64 output tile
  blockDim.x = 4 * WARP_SIZE;
  blockDim.y = 4;

  gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
  gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

  if (gridDim.y >= MAX_GRID_DIMENSION)
  {
    // Quick fix
    gridDim.x *= 3;
    gridDim.y = gridDim.y / 3;
  }

  float alpha = 1.0f;
  float beta = 0.0f;

  printf("Launching a kernel with grid dim: %dx%d and block dimension of (%dx%d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

  if (gridDim.x >= MAX_GRID_DIMENSION || gridDim.y >= MAX_GRID_DIMENSION)
  {
    printf("Invalid grid dimensions.\n");
    return 1;
  }

  // Start time measurement
  cudaEventRecord(start_time);

  printf("WMMA kernel launch?\n");
  apply_filter_GEMM<<<gridDim, blockDim>>>(a_fp16, b_fp16_r, c_wmma_r, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
  apply_filter_GEMM<<<gridDim, blockDim>>>(a_fp16, b_fp16_g, c_wmma_g, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
  apply_filter_GEMM<<<gridDim, blockDim>>>(a_fp16, b_fp16_b, c_wmma_b, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
  cudaErrCheck(cudaDeviceSynchronize()); // ? Required?
  printf("WMMA kernel launch!\n");

  // End time measurement
  cudaEventRecord(end_time);

  // Check for error
  cudaError_t error = cudaPeekAtLastError();
  if (error)
  {
    fprintf(stderr, "Error after kernel launch!: %s\n", cudaGetErrorString(error));
  }

  printf("Copying to host?\n");
  // We only copy over the stuff we need, which is DESIRED_M * DESIRED_N
  cudaErrCheck(cudaMemcpy(c_host_wmma_r, c_wmma_r, DESIRED_M * DESIRED_N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(c_host_wmma_g, c_wmma_g, DESIRED_M * DESIRED_N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(c_host_wmma_b, c_wmma_b, DESIRED_M * DESIRED_N * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Copying to host!\n");

  // numberOfFiltersUsed * image->width * image->height == DESIRED_M * DESIRED_N
  pixel *finalImagesRawData = (pixel *)malloc(DESIRED_M * DESIRED_N * sizeof(pixel));
  if (DESIRED_M != numberOfFiltersUsed)
  {
    printf("Invalid DESIRED_M. Aborting.\n");
    return 1;
  }
  if (DESIRED_N != image->width * image->height)
  {
    printf("Invalid DESIRED_N. Aborting.\n");
    return 1;
  }

  for (int m = 0; m < DESIRED_M; m++)
  {
    for (int n = 0; n < DESIRED_N; n++)
    {
      unsigned char r = (unsigned char)c_host_wmma_r[m * DESIRED_N + n];
      unsigned char g = (unsigned char)c_host_wmma_g[m * DESIRED_N + n];
      unsigned char b = (unsigned char)c_host_wmma_b[m * DESIRED_N + n];
      finalImagesRawData[m * DESIRED_N + n] = (pixel){.b = b, .g = g, .r = r};
    }
  }

  // Blocks CPU execution until end_time is recorded
  cudaEventSynchronize(end_time);

  float spentTime = 0.0;
  cudaEventElapsedTime(&spentTime, start_time, end_time);
  printf("Time spent: %.3f seconds\n", spentTime / 1000);

  cudaEventDestroy(start_time);
  cudaEventDestroy(end_time);

  // Check for error
  error = cudaPeekAtLastError();
  if (error)
  {
    fprintf(stderr, "A CUDA error has occurred while cracking: %s\n", cudaGetErrorString(error));
  }

  //Write the image back to disk
  // if (saveBmpImage(image, output) != 0)
  // {
  //   fprintf(stderr, "Could not save output to '%s'!\n", output);
  //   freeBmpImage(image);
  //   error_exit(&input, &output);
  // };
  for (int i = 0; i < numberOfFiltersUsed; i++)
  {
    char *outputFilename = (char *)calloc(11, sizeof(char));
    memcpy(image->rawdata, finalImagesRawData + i * DESIRED_N, DESIRED_N * sizeof(pixel));
    sprintf(outputFilename, "img_%d.bmp", filterIndexes[i]);
    if (saveBmpImage(image, outputFilename) != 0)
    {
      fprintf(stderr, "Could not save output to '%s'!\n", outputFilename);
      freeBmpImage(image);
      error_exit(&input, &output);
    };
  }

  graceful_exit(&input, &output);
};
