COMPILE_FLAGS := -O3 -std=c++11
LINK_LIBS :=
BINARY := crack
SOURCE_DIR := src

HEADER_FILES := $(shell find $(SOURCEDIR) -name '*.hpp')
HEADER_FILES += $(shell find $(SOURCEDIR) -name '*.cuh')
CPP_SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cpp')
CUDA_SOURCE_FILES += $(shell find $(SOURCEDIR) -name '*.cu')
OBJECT_FILES := $(patsubst %.cpp,%.o,$(CPP_SOURCE_FILES))
OBJECT_FILES += $(patsubst %.cu,%.o,$(CUDA_SOURCE_FILES))

.PHONY: build
build: $(BINARY)

# Compile CUDA
# "-dc" indicates "device compile", for "device code linking"
%.o: %.cu $(HEADER_FILES)
	nvcc $(COMPILE_FLAGS) -dc -o $@ $<

# Compile C++
%.o: %.cpp $(HEADER_FILES)
	g++ $(COMPILE_FLAGS) -c -o $@ $<

# Link
# "-rdc" indicates "create relocatable device code", for "device code linking"
$(BINARY): $(OBJECT_FILES)
	nvcc -rdc=true -o $@ $^ $(LINK_LIBS)

.PHONY: clean
clean:
	find . -type f -name "*.o" -delete
	rm -f $(BINARY)
