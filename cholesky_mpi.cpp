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

using namespace std;
using namespace Eigen;

void potrf(void *buffers[], void *cl_arg) { 
    double *A= (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    int nx = STARPU_MATRIX_GET_NY(buffers[0]);
    int ny = STARPU_MATRIX_GET_NX(buffers[0]);
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', nx, A, ny);
}
struct starpu_codelet potrf_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { potrf, NULL },
    .nbuffers = 1,
    .modes = { STARPU_RW }
};

void trsm(void *buffers[], void *cl_arg) {
    double *A0= (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    double *A1= (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    int nx = STARPU_MATRIX_GET_NY(buffers[0]);
    int ny = STARPU_MATRIX_GET_NX(buffers[0]);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, ny, ny, 1.0, A0, nx, A1, nx);
}
struct starpu_codelet trsm_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { trsm, NULL },
    .nbuffers = 2,
    .modes = { STARPU_R, STARPU_RW }
};

void syrk(void *buffers[], void *cl_arg) { 
    double *A0= (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    double *A1= (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    int nx = STARPU_MATRIX_GET_NY(buffers[0]);
    int ny = STARPU_MATRIX_GET_NX(buffers[0]);
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, nx, nx, -1.0, A0, nx, 1.0, A1, nx);
}
struct starpu_codelet syrk_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { syrk, NULL },
    .nbuffers = 2,
    .modes = { STARPU_R, STARPU_RW }
};

void gemm(void *buffers[], void *cl_arg) {
    double *A0= (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    double *A1= (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    double *A2= (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    int nx = STARPU_MATRIX_GET_NY(buffers[0]);
    int ny = STARPU_MATRIX_GET_NX(buffers[0]);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nx, nx, ny, -1.0,A0, nx, A1, nx, 1.0, A2, nx);
  }
struct starpu_codelet gemm_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { gemm, NULL },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_RW }
};

void cholesky(const int block_size, const int num_blocks, const int rank, const int size, const int test, const int nrow, const int ncol, const bool prune) {
    auto val = [&](int i, int j) { return  1.0/(double)((i-j)*(i-j)+1.0); };
    vector<MatrixXd*> blocks(num_blocks*num_blocks);
    vector<starpu_data_handle_t> dataA(num_blocks*num_blocks);
    auto block_2_rank = [&](int i, int j){return (i % nrow) * ncol + j % ncol;};
    const int ncores = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
    
    {
        size_t ntasks = 0;
        const double start_loop_overhead = starpu_timing_now();
        for (int kk = 0; kk < num_blocks; ++kk) {
            for (int ii = kk+1; ii < num_blocks; ++ii) {
                for (int jj = kk+1; jj < num_blocks; ++jj) {
                    if (jj <= ii) ntasks++;
                }
            }
        }
        const double end_loop_overhead = starpu_timing_now();
        const double t_overhead = (end_loop_overhead - start_loop_overhead)/1e6;
        printf("++++ntasks=%zd,loop_overhead_time=%e\n", ntasks, t_overhead);
    }

    for (int ii=0; ii<num_blocks; ii++) {
        for (int jj=0; jj<num_blocks; jj++) {
            int mpi_rank = block_2_rank(ii,jj);
            blocks[ii+jj*num_blocks] = new MatrixXd();
            if (mpi_rank == rank) {
                auto val_block = [&](int i, int j) { return val(ii*block_size+i,jj*block_size+j); };
                *blocks[ii+jj*num_blocks] = MatrixXd::NullaryExpr(block_size, block_size, val_block);
            }
        }
    }

    size_t num_pruned = 0;
    starpu_mpi_barrier(MPI_COMM_WORLD);
    double start = starpu_timing_now();

    for (int ii=0; ii<num_blocks; ii++) {
        for (int jj=0; jj<num_blocks; jj++) {
            int mpi_rank = block_2_rank(ii,jj);
            if (mpi_rank == rank) {
                starpu_matrix_data_register(&dataA[ii+jj*num_blocks], STARPU_MAIN_RAM, (uintptr_t)blocks[ii+jj*num_blocks]->data(), block_size, block_size, block_size, sizeof(double));
            } else {
                starpu_matrix_data_register(&dataA[ii+jj*num_blocks], -1, (uintptr_t)NULL, block_size, block_size, block_size, sizeof(double));
            }
            if (dataA[ii+jj*num_blocks]) {
                starpu_mpi_data_register(dataA[ii+jj*num_blocks], ii+jj*num_blocks, mpi_rank);
            }
        }
    }

    for (int kk = 0; kk < num_blocks; ++kk) {
        // POTRF
        if( (!prune) || block_2_rank(kk,kk) == rank) {
            starpu_mpi_task_insert(MPI_COMM_WORLD,&potrf_cl,
                STARPU_RW, dataA[kk+kk*num_blocks],
            0);
        } else {
            num_pruned++;
        }

        for (int ii = kk+1; ii < num_blocks; ++ii) {
            // TRSM
            if( (!prune) || block_2_rank(kk,kk) == rank || block_2_rank(ii,kk) == rank) {
                starpu_mpi_task_insert(MPI_COMM_WORLD,&trsm_cl,
                    STARPU_R,  dataA[kk+kk*num_blocks],
                    STARPU_RW, dataA[ii+kk*num_blocks],
                0);
            } else {
                num_pruned++;
            }
            starpu_mpi_cache_flush(MPI_COMM_WORLD, dataA[kk+kk*num_blocks]);

            // SYRK
            if( (!prune) || block_2_rank(ii,kk) == rank ) {
                starpu_mpi_task_insert(MPI_COMM_WORLD,&syrk_cl, 
                    STARPU_R,  dataA[ii+kk*num_blocks],
                    STARPU_RW, dataA[ii+ii*num_blocks],
                0);
            } else {
                num_pruned++;
            }

            for (int jj = kk+1; jj < ii; ++jj) {
                // GEMM
                if( (!prune) || block_2_rank(ii,kk) == rank || block_2_rank(jj,kk) == rank || block_2_rank(ii,jj) == rank) {
                    starpu_mpi_task_insert(MPI_COMM_WORLD,&gemm_cl,
                        STARPU_R,  dataA[ii+kk*num_blocks],
                        STARPU_R,  dataA[jj+kk*num_blocks],
                        STARPU_RW, dataA[ii+jj*num_blocks],
                    0);
                } else {
                    num_pruned++;
                }
            }
            starpu_mpi_cache_flush(MPI_COMM_WORLD, dataA[ii+kk*num_blocks]);
        }
    }

    double end_insertion = starpu_timing_now();
    starpu_task_wait_for_all();
    starpu_mpi_barrier(MPI_COMM_WORLD);
    double end = starpu_timing_now();

    int matrix_size = block_size * num_blocks;
    // Makes grep/import to excel easier ; just do
    // cat output | grep -P '\[0\]\>\>\>\>'
    // to extract rank 0 info
    printf(">>>>test rank nranks ncores matrix_size block_size num_blocks total_time insertion_time prune num_pruned\n");
    printf("[%d]>>>>chol_starpu %d %d %d %d %d %d %e %e %d %zd\n",rank,rank,size,ncores,matrix_size,block_size,num_blocks,(end-start)/1e6,(end_insertion-start)/1e6,prune,num_pruned);

    for (int ii=0; ii<num_blocks; ii++) {
        for (int jj=0; jj<num_blocks; jj++) {
            starpu_data_unregister(dataA[ii+jj*num_blocks]); 
        }
    }

    if (test) {
        printf("Testing...\n");
        for (int ii=0; ii<num_blocks; ii++) {
            for (int jj=0; jj<num_blocks; jj++) {
                if (jj<=ii)  {
                    int mpi_rank = block_2_rank(ii,jj);
                    if (rank == 0 && rank != mpi_rank) {
                        blocks[ii+jj*num_blocks] = new MatrixXd(block_size, block_size);
                        MPI_Recv(blocks[ii+jj*num_blocks]->data(), block_size*block_size, MPI_DOUBLE, mpi_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    } else if (rank == mpi_rank && rank != 0) {
                        MPI_Send(blocks[ii+jj*num_blocks]->data(), block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }
        if (rank==0) {
            MatrixXd B = MatrixXd::NullaryExpr(block_size*num_blocks,block_size*num_blocks, val);
            MatrixXd L = MatrixXd::Zero(block_size*num_blocks,block_size*num_blocks);
            for (int ii=0; ii<num_blocks; ii++) {
                for (int jj=0; jj<num_blocks; jj++) {
                    if (jj<=ii)  {
                        L.block(ii*block_size,jj*block_size,block_size,block_size)=*blocks[ii+jj*num_blocks];
                    }
                }
            }
            auto L1=L.triangularView<Lower>();
            VectorXd x = VectorXd::Random(block_size * num_blocks);
            VectorXd b = B*x;
            VectorXd bref = b;
            L1.solveInPlace(b);
            L1.transpose().solveInPlace(b);
            double error = (b - x).norm() / x.norm();
            printf("\nError solve: %e\n\n", error);
        }
    }
}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_SERIALIZED;
    int prov = -1;
    MPI_Init_thread(NULL, NULL, req, &prov);
    starpu_mpi_init_conf(&argc, &argv, 0, MPI_COMM_WORLD, NULL);
    int rank, size;
    starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
    starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
    if (rank==0) {
        printf("Running on %d CPU cores per rank,", starpu_worker_get_count_by_type(STARPU_CPU_WORKER));
        printf("and %d ranks in total\n", size);
    }
    int block_size=10;
    int num_blocks=1;
    int test=0;
    int nrow=1;
    int ncol=1;
    bool prune = true;

    if (argc >= 2)
    {
        block_size = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_blocks = atoi(argv[2]);
    }

    if (argc >= 4) {
        test = atoi(argv[3]);
    }

    if (argc >= 6) {
        nrow = atoi(argv[4]);
        ncol = atoi(argv[5]);
    }

    if (argc >= 7) {
        prune = atoi(argv[6]);
    }

    assert(nrow * ncol == size);
    printf("Usage: ./cholesky_mpi block_size num_blocks test nrow ncol prune\n");
    printf("block_size,%d\n",block_size);
    printf("num_blocks,%d\n",num_blocks);
    printf("test,%d\n",test);
    printf("nprocs_row,%d\n",nrow);
    printf("nprocs_col,%d\n",ncol);
    printf("prune,%d\n",prune);
    cholesky(block_size, num_blocks, rank, size, test, nrow, ncol, prune);
    starpu_mpi_shutdown();
    MPI_Finalize();
    return 0;
}
