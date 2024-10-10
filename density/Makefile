CC = gcc
CFLAGS = -Iinclude -lm -Wall -Ofast -march=native -fopenmp -DTHREADS=4

SRC_DIR = src
BUILD_DIR = build

all: build/main.o build/particles.o build/grid.o build/density.o
	$(CC) $(CFLAGS) build/main.o build/particles.o build/grid.o build/density.o -o build/main

build/main.o: src/main.c
	$(CC) $(CFLAGS) -c src/main.c -o build/main.o

build/particles.o: src/particles.c
	$(CC) $(CFLAGS) -c src/particles.c -o build/particles.o

build/grid.o: src/grid.c
	$(CC) $(CFLAGS) -c src/grid.c -o build/grid.o

build/density.o: src/density.c
	$(CC) $(CFLAGS) -c src/density.c -o build/density.o

clean:
	rm -rf build/*
