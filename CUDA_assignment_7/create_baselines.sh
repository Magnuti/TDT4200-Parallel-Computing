#!/bin/sh
for k in 0 1 2 3 4
do
    for i in 1 10
    do
        ./main -k $k -i $i before.bmp "task3_${k}_${i}.bmp"
    done
done
