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

I did not change the shared memory solution to convolve all images in one run. Instead I wrote a simple shell script to run the main file multiple times. The result is the same.

### Solution
First, we assume that all filters have the same dimension (3x3 in this case).

The solution is based on [this paper from NVIDIA](https://arxiv.org/pdf/1410.0759.pdf), [this blog from NVIDIA](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/) and on the example code on the blog's [GitHub repository by NVIDIA](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu).

Instead of doing the convolution on all three RGB channels to get one output channel, we perform separate convolutions for the three RGB channels. Why? Because the output image should not be grayscale. Even though this is the way "normal" convolutions are applied (e.g. in CNNs), it is different in this task.

The image passed through the `im2col()` function and the filters is simply flattened out and put in a common array with the `buildFilterArray()` function.

Now we put the filters and images into matrices. The filter goes into the `MxK` matrix, while the image goes into the `KxN` matrix. We can now multiply them with `wmma`. Unfortunately, `wmma` requires the matrix dimensions to be multiplies of 16. We solve this by adding padding in the form of 0s to both matrices. By adding e.g. 7 0s to the rows of matrix A, and 7 0s to the columns of matrix B, the result of the matrix multiplication will still be the same. E.g. $a*x+b*y+c*z+0*0+0*0=a*x+b*y+c*z$.

Desired matrix dimensions
* `mxk` 5x9
* `kxn` 9x9336000 (4000*2334 image)
* Resulting in an 5x9336000 matrix

Actual matrix dimensions
* `mxk` 16x16
* `kxn` 16x9336000
* Resulting in an 16x9336000 matrix

Therefore, we need a padding of 11 0s for the `mxk` matrix' columns and 7 0s for its rows. Likewise, we need a padding of 7 0s for the `kxn` matrix' columns.

In the resulting matrix, we only use the 5x9336000 submatrix, not the entire 16x9336000 matrix.

### Discussion

As you can see on the output images they are not correct. Unfortunately, I did not have time to find the cause of this bug as I started too late on the assignment... Nevertheless, I hope that you are able to understand my attempt on a solution to this GEMM problem, and hopefully you spot the bug rightaway!

Could the 0 padding be removed to get a more efficient matrix multiplication? Perhaps and hopefully? As far as I understood `wmma`, the dimensions must be multiplies of 16, leading to a lot of wasted computation as seen in this case. I would like to see a solution which does not use padding!
