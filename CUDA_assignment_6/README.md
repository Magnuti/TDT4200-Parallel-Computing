# CUDA Assignment 6

This assignment is based on `Assignment_3`, only here we use CUDA to parallelize the task.

## How to run

```
make main
./main -k [kernel] -i [iterations] before.bmp [outputname.bmp]
```

## Baseline

```
./main -k 2 -i 1 before.bmp baseline.bmp
./main -k 2 -i 10 before.bmp baseline_10_i.bmp
./main -k 2 -i 10 before.bmp baseline5x5.bmp
```
The result is stored in `baseline.bmp`, `baseline_10_i.bmp` and `baseline5x5.bmp`.

## Results with `-k 2`
### CPU serial versions
1 iteration: 0.166 seconds  
10 iterations: 1.636 seconds  
100 iterations: 16.190 seconds  

### Task 1 - a single thread on the core (serial version)
One threadblock and 1 thread:
```
applyFilter_CUDA_Kernel<<<1, 1>>>(...
```
1 iteration: 7.833 seconds
10 iterations: 78.162 seconds
10 iterations: 781.211 seconds

### Task 4
`blockSize (16, 16)` and `gridSize (250, 145)`
```
applyFilter_CUDA_Kernel<<<gridSize, blockSize>>>(...
```

1 iteration: 0.001 seconds
10 iterations: 0.008 seconds
100 iterations: 0.076 seconds

As we can see, the parallelized GPU version is around $16.190/0.076=213$ times faster than the serial CPU version, and it is ($781.211/0.076=10279$) around 10.000 times faster than the serial GPU version.
