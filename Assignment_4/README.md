# Assignment 4 - Pthreads and OpenMP

The assignemnt problem is the same as for Assignment 2, only here we use Pthreads/OpenMP instead of OpenMPI.



## Pthreads

Getting started:
```
$ cd PS3b-pthreads-password-cracker-task/
$ make
```

We can change the number of threads by editing line 174: `#define MAX_THREADS 1`

`$ ./crack -i data/shadow/sha512-1word -d data/dict/12dicts/2of5core.txt -l 1`
Max symbols: 1  
Symbol separator: ""  
Shadow file: data/shadow/sha512-1word  
Dictionary file: data/dict/12dicts/2of5core.txt  
Read 4690 words from dictionary file.  

### Results
1 thread
Total duration: 10.284s  
Total attempts: 6420  
Total attempts per second: 624.298  

2
Total duration: 5.381s
Total attempts: 6420
Total attempts per second: 1193.171

4 threads
Total duration: 2.881s
Total attempts: 6420
Total attempts per second: 2228.363

8
Total duration: 1.814s  
Total attempts: 6420  
Total attempts per second: 3539.534  

16
Total duration: 1.288s
Total attempts: 6420
Total attempts per second: 4985.706

24
Total duration: 1.024s  
Total attempts: 6420  
Total attempts per second: 6268.341  

32
Total duration: 1.225s
Total attempts: 6420
Total attempts per second: 5242.107

64 
Total duration: 1.087s  
Total attempts: 6420  
Total attempts per second: 5904.754  

128
Total duration: 1.081s  
Total attempts: 6420  
Total attempts per second: 5941.478  

As we can see, the lowest runtime is gained at around 24 threads. This computational job was executed on a Ryzen 3900X processor, which has 24 logical processors, thus we can see why 24 threads yielded the lowest runtime.

## OpenMP

