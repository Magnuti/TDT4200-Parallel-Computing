# Assignment 2 - MPI

## Task 1

### A)

`$ mpirun -n 1 crack -i data/shadow/sha512-1word -d data/dict/12dicts/2of5core.txt -l 1`

Workers: 1  
Max symbols: 1  
Symbol separator: ""  
Shadow file: data/shadow/sha512-1word  
Dictionary file: data/dict/12dicts/2of5core.txt  
Read 4690 words from dictionary file.

Overview:  
Total duration: 9.652s  
Total attempts: 6420  
Total attempts per second: 665.127  
Skipped: 18  
Successful: 4  
Failed: 0

`$ mpirun -n 1 crack -i data/shadow/sha512-2alnum -d data/dict/alnum.txt -l 2`

Workers: 1  
Max symbols: 2  
Symbol separator: ""  
Shadow file: data/shadow/sha512-2alnum  
Dictionary file: data/dict/alnum.txt  
Read 62 words from dictionary file.

Overview:  
Total duration: 9.631s  
Total attempts: 6396  
Total attempts per second: 664.076  
Skipped: 18  
Successful: 4  
Failed: 0  

`$ mpirun -n 1 crack -i data/shadow/sha512-common -d data/dict/seclists/10-million-password-list-top-10000.txt -l 1`

Workers: 1  
Max symbols: 1  
Symbol separator: ""  
Shadow file: data/shadow/sha512-common  
Dictionary file: data/dict/seclists/10-million-password-list-top-10000.txt  
Read 10000 words from dictionary file.

Overview:  
Total duration: 1.852s  
Total attempts: 1231  
Total attempts per second: 664.609  
Skipped: 18  
Successful: 4  
Failed: 0  

### B)

With 2 characters, we have a total of $62^2=3844$ possible password combinations. We have around 664 attempts per second. Worst-case for this it $\frac{3844}{664}=5.79$ seconds per password and an average of 2.89 seconds per password. With 8 characters, we have a total of $62^8=2.183*10^{14}$ possible password combinations. Therefore, it can take up to $\frac{2.183*10^{14}}{664.66}=3.29*10^{11}$ seconds (over 10000 years) to break a single 8-character password.

There are 4690 words in our dictionary. A 4-word password, with our dictionary, has $4690^4=4.84*10^{14}$ different password combinations. Since we have around 665.127 attempts per second, it could take us $\frac{4.84*10^{14}}{665.127}=7.27*10^{11}$ seconds (over 23000 years) to brute force one 8-character password.

## Task 2

` $ mpirun -n 12 crack -i data/shadow/sha512-1word -d data/dict/12dicts/2of5core.txt -l 1`

Greetings from process 1 of 12  
Greetings from process 2 of 12  
Greetings from process 10 of 12  
Greetings from process 5 of 12  
Greetings from process 7 of 12  
Greetings from process 11 of 12  
Greetings from process 8 of 12  
Greetings from process 9 of 12  
Workers: 12  
Max symbols: 1  
Symbol separator: ""  
Shadow file: data/shadow/sha512-1word  
Greetings from process 3 of 12  
Greetings from process 4 of 12  
Greetings from process 6 of 12  
Dictionary file: data/dict/12dicts/2of5core.txt  
Read 4690 words from dictionary file.  

Overview:  
Total duration: 10.049s  
Total attempts: 6420  
Total attempts per second: 638.892  
Skipped: 18  
Successful: 4  
Failed: 0  

As we can see, the result is pretty much the same when we run one process vs. 12 processes:

* 1 process: 9.652s
* 12 processes: 10.049s



## Task 3
