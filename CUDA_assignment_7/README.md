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

| Iterations | Running time in seconds |
|:-:|:-:|
| 1 | 0.166 |
| 10 | 1.636 |
| 100 | 16.190 |

### Task 1 - a single thread on the core (serial version)
One threadblock and 1 thread:
```
applyFilter_CUDA_Kernel<<<1, 1>>>(...
```

| Iterations | Running time in seconds |
|:-:|:-:|
| 1 | 7.833 |
| 10 | 78.162 |
| 100 | 781.211 |

### Task 4
`blockSize (16, 16)` and `gridSize (250, 145)`
```
applyFilter_CUDA_Kernel<<<gridSize, blockSize>>>(...
```

| Iterations | Running time in seconds |
|:-:|:-:|
| 1 | 0.001 |
| 10 | 0.008 |
| 100 | 0.076 |

As we can see, the parallelized GPU version is around $16.190/0.076=213$ times faster than the serial CPU version, and it is ($781.211/0.076=10279$) around 10.000 times faster than the serial GPU version.

### Task 5 - Shared memory

| Iterations | Running time in seconds |
|:-:|:-:|
| 1 | 0.001 |
| 10 | 0.011 |
| 100 | 0.110 |

As we can see, we actually have a slowdown from global memory. For further improvements, we could share the border pixels which must be accessed in global memory.

### Task 6 - Occupancy
After running the code with the occupancy API code from https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/, this was our output:
```
BlockSizeInt 1024  
minGridSizeInt 40  
gridSizeInt 9118  
The grid has thread blocks of dimension (32 width * 32 height)  
Launching a grid of dimension (125 width * 73 height)  
Launched blocks of size 1024=>(32x32). Theoretical occupancy: 1.000000  
```

Since our image is $4000x2334$, and we have one thread per pixel, we need $4000x2334=9 336 000$ threads. The result from the occupancy API gives us 9118 blocks of size 1024. Since $9118*1024=9 336 832$, have 832 spare threads.
