#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>

// #include <cuda_runtime_api.h> // ? required??
#include <mma.h> // CUDA WMMA API

using namespace nvcuda; // C++ stuff

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

// TODO remove one of cudaErr.. or cudaError..
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
// float const filterFactors[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0 / 256.0}; // TODO ?

const unsigned int MAX_GRID_DIMENSION = 65535;

const unsigned int numberOfChannels = 3;

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
const unsigned int DESIRED_K = 27;                  // 3*3*3, 3x3 filters with 3 channels
const unsigned int DESIRED_N = 4000 * 2334;         // 4000x2334 image

// Must be multiplies of 16
const int MATRIX_M = 16;        // I want 5 here (5 filters)
const int MATRIX_K = 32;        // I want 27 here (3x3 filter values * 3 channels)
const int MATRIX_N = DESIRED_N; // It is evenly divisible by 16

// im2col and col2im taken from https://github.com/pluskid/Mocha.jl/blob/master/deps/im2col.cpp#L7
// They only work with one channel (the RGB channel is "one" for the pixel struct).
// Furthemore, we assume that we use a valid padding (input dim = output dim) and 1 stride.

void im2col(pixel *img, unsigned char *col, int width, int height, int filterDim)
{
  // The dimension of the col array is im_width*im_height*filter_h*filter_w*numberOfChannels
  int kernel_h = filterDim,  // Assuming square kernel
      kernel_w = filterDim,  // Assuming square kernel
      pad_h = filterDim / 2, // Same padding
      pad_w = filterDim / 2; // Same padding
  int height_col = (height + 2 * pad_h - kernel_h) + 1;
  int width_col = (width + 2 * pad_w - kernel_w) + 1;
  int channels_col = kernel_h * kernel_w;

  for (int color = 0; color < 3; color++)
  {
    unsigned int offset = color * width * height * filterDim * filterDim;
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
          int index_col = offset + (c * height_col + h) * width_col + w;
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
}

void buildFilterArray(char *array)
{
  // We duplicate the filters 3 times, one for each channel
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
        unsigned int temp_j = j % (filterDim * filterDim);
        array[arrayIndex] = filters[filterIndex][temp_j];
      }
      else
      {
        array[arrayIndex] = 0;
      }
    }
  }
}

void buildImageArray(unsigned char *outputCol, unsigned char *inputCol, unsigned int intputColLength)
{
  // The image is of dimension kxn
  // This function adds the 5 0s to the colums
  memcpy(outputCol, inputCol, intputColLength);
  for (int k = 0; k < MATRIX_K - DESIRED_K; k++)
  {
    for (int n = 0; n < MATRIX_N; n++)
    {
      outputCol[intputColLength + k * MATRIX_N + n] = 0;
    }
  }
}

// void seperateImageChannels(unsigned char *array, bmpImage *image)
// {
//   // Builds an array consisting of red1, red2, ..., green1, green2, ..., blue1, blue2, blue3, ...
//   for (int h = 0; h < image->height; h++)
//   {
//     for (int w = 0; w < image->width; w++)
//     {
//       array[h * image->width + w] = image->data[h][w].r;
//     }
//   }

//   for (int h = 0; h < image->height; h++)
//   {
//     for (int w = 0; w < image->width; w++)
//     {
//       int offset = image->height * image->width;
//       array[offset + h * image->width + w] = image->data[h][w].g;
//     }
//   }

//   for (int h = 0; h < image->height; h++)
//   {
//     for (int w = 0; w < image->width; w++)
//     {
//       int offset = image->height * image->width * 2;
//       array[offset + h * image->width + w] = image->data[h][w].b;
//     }
//   }
// }

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

  // Load the inputs
  // wmma::load_matrix_sync(a_frag, a, 16);
  // wmma::load_matrix_sync(b_frag, b, 16);

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

__global__ void convertFp32ToFp16(half *out, float *in, int n)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n)
  {
    out[idx] = in[idx];
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
  // const float usedFilterFactor = filterFactors[filterIndexes[0]]; // TODO

  int *filterCol = (int *)malloc(MATRIX_M * MATRIX_K * sizeof(int));
  unsigned int tempImageColLength = image->width * image->height * filterDim * filterDim * numberOfChannels;
  unsigned char *tempImageCol = (unsigned char *)malloc(tempImageColLength * sizeof(unsigned char));
  im2col(image->rawdata, tempImageCol, image->width, image->height, filterDim);

  unsigned char *imageColChar = (unsigned char *)malloc(MATRIX_K * MATRIX_N * sizeof(unsigned char));
  buildImageArray(imageColChar, tempImageCol, tempImageColLength);

  free(tempImageCol);

  float *imageCol = (float *)malloc(MATRIX_K * MATRIX_N * sizeof(float));
  for (int i = 0; i < MATRIX_K * MATRIX_N; i++)
  {
    imageCol[i] = (float)imageColChar[i];
  }

  free(imageColChar);

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

  // int image_size = image->width * image->height * sizeof(pixel);

  // We could also made all filters __device__ available, but it is simple to copy over only the needed one
  // pixel *d_image_rawdata, *d_process_image_rawdata;
  // int *d_filters;

  // cudaMalloc((void **)&d_image_rawdata, image_size); // ! * numberOfFilters ?
  // cudaMalloc((void **)&d_process_image_rawdata, image_size * numberOfFiltersUsed);
  // cudaMalloc((void **)&d_filters, filter_size * numberOfFiltersUsed);

  // cudaMemcpy(d_image_rawdata, image->rawdata, image_size, cudaMemcpyHostToDevice);

  // We allocate one thread per pixel
  // gridSize and blockSize inspired from Section 2.2. in the CUDA Programming Guide
  // dim3 blockSize(BLOCK_DIMENSION, BLOCK_DIMENSION); // Threads per block
  // printf("The grid has thread blocks of dimension (%d width * %d height)\n", blockSize.x, blockSize.y);

  // We may need to add 1 extra block to width or height if the image's dimensions are not evenly divided by the block's dimension
  // int extraWidth = 0;
  // int extraHeight = 0;
  //
  // if (image->width % blockSize.x != 0)
  // {
  //   extraWidth = 1;
  // }
  // if (image->height % blockSize.y != 0)
  // {
  //   extraHeight = 1;
  // }
  // dim3 gridSize(image->width / blockSize.x + extraWidth, image->height / blockSize.y + extraHeight); // Number of blocks
  // printf("Launching a grid of dimension (%d width * %d height)\n", image->width / blockSize.x + extraWidth, image->height / blockSize.y + extraHeight);

  // TODO fix comments
  // All taken from https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu
  float *a_fp32;      // Filter temp
  float *b_fp32;      // Image temp
  half *a_fp16;       // Filter
  half *b_fp16;       // Image array
  float *c_wmma;      // Device answer array
  float *c_host_wmma; // Host answer array

  cudaErrCheck(cudaMalloc((void **)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
  cudaErrCheck(cudaMalloc((void **)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void **)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
  cudaErrCheck(cudaMalloc((void **)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

  cudaErrCheck(cudaMalloc((void **)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

  c_host_wmma = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));

  // TODO I think I need to copy the arrays to device memory, curand did this by itself I think.
  // TODO so I must do it explicitly.

  // curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  // curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

  // TODO fill a_fp32 and b_fp32 with numbers instead of curandGenerator stuff
  cudaErrCheck(cudaMemcpy(a_fp32, filterCol, MATRIX_M * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(b_fp32, imageCol, MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));

  // curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
  // curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
  // printf("Float to half ?\n");
  printf("Float to half kernel launch with grid dim %d, block dim %d\n", (MATRIX_M * MATRIX_K + 255) / 256, 256);
  convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 255) / 256, 256>>>(a_fp16, a_fp32, MATRIX_M * MATRIX_K);
  printf("Float to half kernel launch with grid dim %d, block dim %d\n", (MATRIX_K * MATRIX_N + 255) / 256, 256);
  convertFp32ToFp16<<<(MATRIX_K * MATRIX_N + 255) / 256, 256>>>(b_fp16, b_fp32, MATRIX_K * MATRIX_N);
  // printf("Float to half done!\n");
  cudaErrCheck(cudaDeviceSynchronize()); // ? Required?

  for (int i = 0; i < MATRIX_M * MATRIX_N; i++)
  {
    c_host_wmma[i] = 0.0f;
  }
  cudaErrCheck(cudaMemcpy(c_wmma, c_host_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));

  // curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));

  // curandErrCheck(curandDestroyGenerator(gen));

  dim3 gridDim;
  dim3 blockDim;

  // blockDim.x must be a multple of warpSize
  // 128x4 means we have 16 warps and a block computes a 64x64 output tile
  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
  gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

  // ! 145875 in x direction...
  if (gridDim.x >= MAX_GRID_DIMENSION || gridDim.y >= MAX_GRID_DIMENSION)
  {
    printf("Invalid grid dimensions.\n");
    return 1;
  }

  // TODO remove
  float alpha = 1.0f;
  float beta = 1.0f;

  printf("Launching a kernel with grid dim: %dx%d and block dimension of (%dx%d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

  // Start time measurement
  cudaEventRecord(start_time);

  // for (unsigned int i = 0; i < iterations; i++)
  // {
  // int sharedMemoryUsedPerBlock = numberOfFiltersUsed * usedFilterDimension * usedFilterDimension * sizeof(int) + BLOCK_DIMENSION * BLOCK_DIMENSION * sizeof(pixel);
  // apply_filter<<<gridSize, blockSize, sharedMemoryUsedPerBlock>>>(
  //     d_process_image_rawdata, // Out
  //     d_image_rawdata,         // In
  //     image->width,
  //     image->height,
  //     // filters[filterIndex],
  //     d_filters,
  //     numberOfFiltersUsed,
  //     usedFilterDimension,
  //     filter_size,
  //     usedFilterFactor);

  // ? Do I need to pass in WMMA_M, _n, _k?
  printf("WMMA kernel launch?\n");
  apply_filter_GEMM<<<gridDim, blockDim>>>(a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
  cudaErrCheck(cudaDeviceSynchronize()); // ? Required?
  printf("WMMA kernel launch!\n");
  // swapImage(&processImage, &image);
  // swapImageRawdata(&d_process_image_rawdata, &d_image_rawdata);
  // }

  // Check for error
  cudaError_t error = cudaPeekAtLastError();
  if (error)
  {
    fprintf(stderr, "Error after kernel launch!: %s\n", cudaGetErrorString(error));
  }

  // End time measurement
  cudaEventRecord(end_time);

  cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

  // TODO clean this lol
  pixel *finalImageRawData0 = (pixel *)malloc(image->width * image->height * sizeof(pixel));
  pixel *finalImageRawData1 = (pixel *)malloc(image->width * image->height * sizeof(pixel));
  pixel *finalImageRawData2 = (pixel *)malloc(image->width * image->height * sizeof(pixel));
  pixel *finalImageRawData3 = (pixel *)malloc(image->width * image->height * sizeof(pixel));
  pixel *finalImageRawData4 = (pixel *)malloc(image->width * image->height * sizeof(pixel));
  for (int m = 0; m < DESIRED_M; m++)
  {
    for (int n = 0; n < DESIRED_N; n++)
    {
      unsigned char value = (unsigned char)(c_host_wmma[m * DESIRED_N + n] / 3); // TODO try to divide by 3
      switch (m)
      {
      case 0:
        finalImageRawData0[n] = (pixel){.b = value, .g = value, .r = value};
        break;
      case 1:
        finalImageRawData1[n] = (pixel){.b = value, .g = value, .r = value};
        break;
      case 2:
        finalImageRawData2[n] = (pixel){.b = value, .g = value, .r = value};
        break;
      case 3:
        finalImageRawData3[n] = (pixel){.b = value, .g = value, .r = value};
        break;
      case 4:
        finalImageRawData4[n] = (pixel){.b = value, .g = value, .r = value};
        break;
      default:
        break;
      }
    }
  }

  // cudaMemcpy(image->rawdata, d_image_rawdata, image_size, cudaMemcpyDeviceToHost);

  // cudaFree(d_image_rawdata);
  // cudaFree(d_process_image_rawdata);

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
  memcpy(image->rawdata, finalImageRawData0, image->width * image->height);
  char outputName[11] = "wmma_x.bmp";
  strcpy(outputName, "wmma_0.bmp");
  if (saveBmpImage(image, outputName) != 0)
  {
    fprintf(stderr, "Could not save output to '%s'!\n", outputName);
    freeBmpImage(image);
    error_exit(&input, &output);
  };

  memcpy(image->rawdata, finalImageRawData1, image->width * image->height);
  strcpy(outputName, "wmma_1.bmp");
  if (saveBmpImage(image, outputName) != 0)
  {
    fprintf(stderr, "Could not save output to '%s'!\n", outputName);
    freeBmpImage(image);
    error_exit(&input, &output);
  };

  memcpy(image->rawdata, finalImageRawData2, image->width * image->height);
  strcpy(outputName, "wmma_2.bmp");
  if (saveBmpImage(image, outputName) != 0)
  {
    fprintf(stderr, "Could not save output to '%s'!\n", outputName);
    freeBmpImage(image);
    error_exit(&input, &output);
  };

  memcpy(image->rawdata, finalImageRawData3, image->width * image->height);
  strcpy(outputName, "wmma_3.bmp");
  if (saveBmpImage(image, outputName) != 0)
  {
    fprintf(stderr, "Could not save output to '%s'!\n", outputName);
    freeBmpImage(image);
    error_exit(&input, &output);
  };

  memcpy(image->rawdata, finalImageRawData4, image->width * image->height);
  strcpy(outputName, "wmma_4.bmp");
  if (saveBmpImage(image, outputName) != 0)
  {
    fprintf(stderr, "Could not save output to '%s'!\n", outputName);
    freeBmpImage(image);
    error_exit(&input, &output);
  };

  graceful_exit(&input, &output);
};
