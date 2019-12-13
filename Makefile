CC       = g++
MPICC    = mpicxx
CFLAGS   = -DEIGEN_USE_LAPACKE --std=c++14 -g -O0 -DEIGEN_NO_DEBUG `pkg-config --cflags starpu-1.3`
INCLUDE  =
LFLAGS   = `pkg-config --libs starpu-1.3` -llapack -lblas -lpthread

.PHONY: clean

cholesky_starpu: cholesky_starpu.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LFLAGS)

cholesky_mpi: cholesky_mpi.cpp
	$(MPICC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LFLAGS)

clean:
	rm -f cholesky_starpu
	rm -f cholesky_mpi
