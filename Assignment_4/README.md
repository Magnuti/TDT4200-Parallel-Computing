# Assignment 4 - Pthreads and OpenMP

The assignemnt problem is the same as for Assignment 2, only here we use Pthreads/OpenMP instead of OpenMPI.



## Pthreads

```
$ cd PS3b-pthreads-password-cracker-task/
$ make
```

We can change the number of threads by editing line 174: `#define MAX_THREADS 1`



## OpenMP

```
$ cd PS3b-openmp-password-cracker-task/
$ make
```

The number of threads can be modified on line 168: `size_t num_threads = 4;`



## Results

`$ ./crack -i data/shadow/sha512-1word -d data/dict/12dicts/2of5core.txt -l 1`
Max symbols: 1  
Symbol separator: ""  
Shadow file: data/shadow/sha512-1word  
Dictionary file: data/dict/12dicts/2of5core.txt  
Read 4690 words from dictionary file.  

| Threads          | 1       | 2      | 4      | 8      |   16   | 24     | 32     | 64     | 128    |
|------------------|---------|--------|--------|--------|:------:|--------|--------|--------|--------|
| Pthreads runtime | 10.284s | 5.381s | 2.881s | 1.814s | 1.288s | 1.024s | 1.225s | 1.087s | 1.081s |
| OpenMP runtime   | 12.081s | 6.236s | 3.684s | 2.241s | 1.181s | 0.940s | 1.147s | 0.970s | 0.937s |

As we can see, the lowest runtime is gained at around 24 threads. This computational job was executed on a Ryzen 3900X processor, which has 24 logical processors, thus we can see why 24 threads yielded the lowest runtime.