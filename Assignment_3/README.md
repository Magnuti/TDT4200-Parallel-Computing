# Assignment 3

## How to run

From Unix or WSL:
```
$ make main
$ mpirun -n [number_of_processes] main -k [kernel_index [0-5]] -i [iterations] before.bmp [output filename e.g. after.bmp]
```

## Baseline

Baseline with the Laplacian 1 kernel for 100 iterations
```
$ mpirun -n 1 main -k 2 -i 100 before.bmp baseline.bmp
```
Time spent: 15.425 seconds
