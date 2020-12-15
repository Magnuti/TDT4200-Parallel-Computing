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
