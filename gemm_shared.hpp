#include <starpu.h>
#include <starpu_mpi.h>
#include <vector>
#include <memory>
#include <iostream>
#ifdef USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <mpi.h>

void gemm(void *buffers[], void *cl_arg) {
    double *A0 = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    double *A1 = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    double *A2 = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    int nx     = STARPU_MATRIX_GET_NY(buffers[0]);
    int ny     = STARPU_MATRIX_GET_NX(buffers[0]);
    assert(nx == ny);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nx, nx, nx, 1.0, A0, nx, A1, nx, 1.0, A2, nx);
  }
struct starpu_codelet gemm_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { gemm, NULL },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_RW }
};

auto val = [&](int i, int j) { return  (double) (i % 37 + j * i % 49);};
