CC = gcc
CFLAGS = -I.
DEPS = bitmap.h
OBJ = main.o bitmap.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

bitmap: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)
	@echo "Build complete!"
# Same as:
# bitmap: main.o bitmap.o
	# gcc -o bitmap main.o bitmap.o
	# @echo "Build complete!"

.PHONY: clean

# Delete all .o files on cleanup
clean:
	@rm -f *.o
