# Assignment 2 - MPI

## Task 1

### A)

`$ mpirun -n 1 crack -i data/shadow/sha512-1word -d data/dict/12dicts/2of5core.txt -l 1`

Max symbols: 1  
Symbol separator: ""  
Shadow file: data/shadow/sha512-1word  
Dictionary file: data/dict/12dicts/2of5core.txt  
Read 4690 words from dictionary file.

Overview:  
Total duration: 0.000s  
Total attempts: 6420  
Total attempts per second: inf  
Skipped: 18  
Successful: 4  
Failed: 0

`$ mpirun -n 1 crack -i data/shadow/sha512-2alnum -d data/dict/alnum.txt -l 2`

Max symbols: 2  
Symbol separator: ""  
Shadow file: data/shadow/sha512-2alnum  
Dictionary file: data/dict/alnum.txt  
Read 62 words from dictionary file.

Overview:  
Total duration: 0.000s  
Total attempts: 6396  
Total attempts per second: inf  
Skipped: 18  
Successful: 4  
Failed: 0  

`$ mpirun -n 1 crack -i data/shadow/sha512-common -d data/dict/seclists/10-million-password-list-top-10000.txt -l 1`

Max symbols: 1  
Symbol separator: ""  
Shadow file: data/shadow/sha512-common  
Dictionary file: data/dict/seclists/10-million-password-list-top-10000.txt  
Read 10000 words from dictionary file.

Overview:  
Total duration: 0.000s  
Total attempts: 1231  
Total attempts per second: inf  
Skipped: 18  
Successful: 4  
Failed: 0  

### B)
It takes `x` seconds to break a two-character output, with 62 entries. 62^2=3844. 62^8=2.183*10^14.

With one process, we have 3844/x=k operations per second.

2.183*10^14 operations divided by k = our answer.

Do the same for a 4 word password.

### C)

## Task 2

## Task 3
