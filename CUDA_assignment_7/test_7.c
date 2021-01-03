#include <stdio.h>
#include <stdlib.h>

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

const unsigned int numberOfChannels = 3;

// Hardcoded selected filters for now
const unsigned int numberOfIndexes = 5;
const unsigned int filterIndexes[] = {0, 1, 2, 3, 4};
const unsigned int filterDim = 3; // Only one filter dimension supported for now

// WMMA stuff from https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// mxk * kxn = mxn
const unsigned int DESIRED_M = numberOfIndexes; // 5 filters
const unsigned int DESIRED_K = 27;              // 3*3*3, 3x3 filters with 3 channels
const unsigned int DESIRED_N = 4000 * 2334 * 3; // 4000x2334 image with 3 channels

// Must be multiplies of 16
const int MATRIX_M = 16;        // I want 5 here (5 filters)
const int MATRIX_K = 32;        // I want 27 here (3x3 filter values * 3 channels)
const int MATRIX_N = DESIRED_N; // It is evenly divisible by 16

void buildFilterArray(char *array)
{
    // We duplicate the filters 3 times, one for each channel
    // The filter is of dimension mxk
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

int main(int argc, char const *argv[])
{
    char *filterArray = malloc(MATRIX_M * MATRIX_K * sizeof(char));
    buildFilterArray(filterArray);
    for (int i = 0; i < MATRIX_M; i++)
    {
        printf("[");
        for (int j = 0; j < MATRIX_K; j++)
        {
            printf("%d, ", filterArray[i * MATRIX_K + j]);
        }
        printf("]\n");
    }

    return 0;
}
