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
### Task 1 - a single thread on the core (serial version)
One threadblock and 1 thread:
```
applyFilter_CUDA_Kernel<<<1, 1>>>(...
```
1 iteration  
Time spent: 7.878 seconds

### Task 2, 3 and 4
`blockSize (16, 16)` and `gridSize (250, 145)`
```
applyFilter_CUDA_Kernel<<<gridSize, blockSize>>>(...
```

1 iteration  
Time spent: 0.001 seconds

10 iterations  
Time spent: 0.008 seconds

