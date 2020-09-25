# Assignment 3

## How to run

From Unix or WSL:
```
$ make main
$ mpirun -n [number_of_processes] ./main -k [kernel_index [0-5]] -i [iterations] before.bmp [output filename e.g. after.bmp]
```

## Baseline

Baseline with the Laplacian 1 kernel for 100 iterations
```
$ mpirun -n 1 ./main -k 2 -i 100 before.bmp baseline.bmp
```
Time spent: 15.425 seconds

Baseline with the 5x5 Gaussian kernel for 100 iterations
```
$ mpirun -n 1 ./main -k 5 -i 100 before.bmp baseline5x5.bmp
```
Time spent: 33.981 seconds

## Task 2

### 2.1

#### 2.1.1
First each process receives neccessary information with `MPI_Bcast` and their rows from `MPI_Scatterv`.

The even number processes swap borders with the `my_rank+1` processes first, then with the `my_rank-1` processes, and vice versa. This communication pattern avoids deadlock.
When the processes have exchanged borders, they apply the convolution to their rows.
The border exhange and convolution applying loops for a given number of iterations.

When the iteration loop is completed, each process sends their processed rows to master with `MPI_Gatherv`. The master rank then assembles the image and saves it.

#### 2.1.2
Since the the Laplacian kernel is 3x3, we only have to send/receive a single row. So, each process sends two rows (upper and lower) and receives two rows.

For a 4000 width image, with each pixel being 3 bytes, we need to transfer 12000 bytes in one row.

So, a non-border process must send and receive 24000 bytes, while a border-process must send and receive 12000 bytes.

Sicne we have 4 processes, we have 3 exhanges (since the uppermost and lowermost process are on the borders). So, 24000*3=72000 bytes. 72000*2=144000 bytes with 2 iterations.

If we include the scattering and gathering, we need to scatter 4000x2334*3 = rougly 27 MB, and gather the same amount.

We ignore the `MPI_Bcast` since it is not much.

#### 2.1.3
With 8 processes, we need 7 exhanges. Following the same calculations from above, we get 24000*7=168000 bytes. 168.000*2=336.000 bytes with 2 iterations.

### 2.2

Measurements for the Laplacian Kernel `k=2`:

| Processes/iterations      | i=1           | i=10           | i=100             |
| ------------------------- |:-------------:|:--------------:| -----------------:|
| 1                         | 0.170 seconds | 1.486 seconds  | 15.425 seconds    |         
| 5                         | 0.049 seconds | 0.334 seconds  | 5.707 seconds     |         
| 10                        | 0.045 seconds | 0.296 seconds  | 2.926 seconds     |

1 processes vs. 100 processes with 100 iterations: 15.425/22.926=5.27. This means that our program runs 5.27 times faster with 10 processes instead of 1 for 100 iterations.

Measurements for the Gaussian Kernel `k=5`:

| Processes/iterations      | i=1           | i=10           | i=100             |
| ------------------------- |:-------------:|:--------------:| -----------------:|
| 1                         | 0.363 seconds | 3.426 seconds  | 33.981 seconds    |         
| 5                         | 0.091 seconds | 0.726 seconds  | 10.797 seconds    |         
| 10                        | 0.085 seconds | 0.692 seconds  | 6.805 seconds     |
