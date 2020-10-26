# CUDA Assignment 5
Password cracker with CUDA. The cracking job is the same as with assignemnt 2 and 4.

Since we are using CUDA, we only use SHA-512 hashes. We do not use the `crypt()` function from the GNU C library.

Example runs `GRID_SIZE 40` and `BLOCK_SIZE 32 * 2`

```
./crack -i data/shadow/sha512-4num -d data/dict/num.txt -l 4
```
Total duration: 0.487s  
Total attempts per second: 39147.138  

```
./crack -i data/shadow/sha512-2alnum -d data/dict/alnum.txt -l 2
```
Total duration: 0.295s  
Total attempts per second: 21655.164  

```
./crack -i data/shadow/sha512-1word -d data/dict/12dicts/2of5core.txt
```
Total duration: 0.363s  
Total attempts per second: 17668.807  

```
./crack -i data/shadow/sha512-2word -d data/dict/12dicts/2of5core.txt -l2 -s" "
```
Total duration: 1546.459s --> around 25 minutes    
Total attempts: 28 805 334  
Total attempts per second: 18626.638  

```
./crack -i data/shadow/sha512-common -d data/dict/seclists/10-million-password-list-top-10000.txt
```
Total duration: 0.247s  
Total attempts per second: 4982.900  

