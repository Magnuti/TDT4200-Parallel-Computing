# CUDA Assignment 6

This assignment is a further improvement of `Assignment_5`.

## How to run

```
make main
./main -k [kernel] -i [iterations] before.bmp [outputname.bmp]
```

## Task C Tensor cores and convolution as GEMM

We use all 5 3x3 filters (filter 0-4).

### Baseline images
The baseline images are generated for kernel 0-4 with 1 and 10 iterations and are stored as `task3_[kernel index]_[iterations].bmp`. For example, `task3_2_10.bmp` shows the image for kernel 2 after 10 iterations.

### Solution
We assume that all filters have the same dimension (3x3 in this case).

We duplicate each filter 3 times because we separate the RGB channels. Why? Such that the `wmma` can be implemented nicely. By doing this, we also create a nice opportunity to use different filters for different channels in the future.

We separate the image into 3 channels, one for each RGB channel.

Now we put the filters and images into matrices. The filter goes into the `MxK` matrix, while the image goes into the `KxN` matrix. We can now multiply them with `wmma`. Unfortunately, `wmma` requires the matrix dimensions to be multiplies of 16. We solve this by adding padding in the form of 0s to both matrices. By adding e.g. 5 0s to the rows of matrix A, and 5 0s to the columns of matrix B, the result of the matrix multiplication will still be the same. E.g. $a*x+b*y+c*z+0*0+0*0=a*x+b*y+c*z$.

Desired matrix dimensions
* `mxk` 5x27
* `kxn` 27x9336000 (4000*2334)
* Resulting in an 5x9336000 matrix

Actual matrix dimensions
* `mxk` 16x32
* `kxn` 32x9336000
* Resulting in an 16x9336000 matrix

Therefore, we need a padding of 11 0s for the `mxk` matrix' columns and 5 0s for its rows. Likewise, we need a padding of 5 0s for the `kxn` matrix' columns.

In the resulting matrix, we only use the 5x9336000 submatrix, not the entire 16x9336000 matrix.
