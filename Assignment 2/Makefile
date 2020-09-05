CC := mpicc
COMPILE_FLAGS := -O3 -std=gnu11
LINK_LIBS := -lcrypt
BINARY := crack
SOURCE_DIR := src

HEADER_FILES := $(shell find $(SOURCEDIR) -name '*.h')
SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.c')
OBJECT_FILES := $(patsubst %.c,%.o,$(SOURCE_FILES))

.PHONY: build
build: $(BINARY)

.PHONY: clean
clean:
	find . -type f -name "*.o" -delete
	rm -f $(BINARY)

# Link
$(BINARY): $(OBJECT_FILES)
	$(CC) -o $@ $^ $(LINK_LIBS)

# Compile
%.o: %.c $(HEADER_FILES)
	$(CC) $(COMPILE_FLAGS) -c -o $@ $<
