CC       = g++
CFLAGS   = -DEIGEN_USE_LAPACKE --std=c++14 -g -O0 -DEIGEN_NO_DEBUG
INCLUDE  = $(pkg-config --cflags starpu-1.3)
LFLAGS   = -llapack -lblas -lpthread $(pkg-config --libs starpu-1.3)

.PHONY: clean

cholesky_starpu: cholesky_starpu.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LFLAGS)

clean:
	rm -f cholesky_starpu
