CC       = icpc
MPICC    = mpiicpc
CFLAGS   = -DUSE_MKL -DEIGEN_USE_MKL_ALL -std=c++14 -g -O3 -I${HOME}/Softwares/hwloc-2.2.0/install/include -I${HOME}/Softwares/starpu-1.3.2/install/include/starpu/1.3 -I${HOME}/Softwares/eigen/ -I${MKLROOT}/include
INCLUDE  =
LFLAGS   = -lpthread -L${HOME}/Softwares/hwloc-2.2.0/install/lib -L${HOME}/Softwares/starpu-1.3.2/install/lib -lstarpumpi-1.3 -lstarpu-1.3 -lhwloc -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

.PHONY: clean all

all: cholesky_starpu cholesky_mpi gemm_2d_mpi

cholesky_starpu: cholesky_starpu.cpp
	$(MPICC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LFLAGS)

cholesky_mpi: cholesky_mpi.cpp
	$(MPICC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LFLAGS)

gemm_2d_mpi: gemm_2d_mpi.cpp
	$(MPICC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LFLAGS)

clean:
	rm -f cholesky_starpu cholesky_mpi gemm_2d_mpi
