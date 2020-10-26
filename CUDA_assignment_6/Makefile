##
# Game of Life Makefile
#
# @file
# @version 0.1

SRC := $(wildcard *.cu)
LIBS := $(wildcard libs/*.c)
OBJ := $(patsubst %.c,%.o,$(SRC))

CC := nvcc 

ifdef DEBUG
FLAGS := -g
else
FLAGS := -O3
endif

.PHONY: clean

main: $(LIBS) $(SRC)
	$(CC) $(FLAGS) $(LIBS) $(SRC) -o main

clean:
	rm -Rf main.o
	rm -Rf main
	rm -Rf libs/bitmap.o

