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

## CPU running time with `-k 2`
1 iteration: 0.166 seconds  
10 iterations: 1.636 seconds  
100 iterations: 16.190 seconds  
