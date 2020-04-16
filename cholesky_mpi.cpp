#include <starpu.h>
#include <starpu_mpi.h>
#include<vector>
#include<memory>
#include<iostream>
#include <cblas.h>
#include <lapacke.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <mpi.h>


#define TAG11(k)	((starpu_tag_t)( (1ULL<<60) | (unsigned long long)(k)))
#define TAG21(k,j)	((starpu_tag_t)(((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG22(k,i,j)	((starpu_tag_t)(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))


using namespace std;
using namespace Eigen;


void potrf(void *buffers[], void *cl_arg) { 

    double *A= (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    int nx = STARPU_MATRIX_GET_NY(buffers[0]);
	int ny = STARPU_MATRIX_GET_NX(buffers[0]);

	//LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A->rows(), A->data(), A->rows());
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', nx, A, ny);
    //Map<MatrixXd> tt(A, nx, nx);
    //cout<<"POTRF: \n"<<tt<<"\n";
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
    //Map<MatrixXd> tt(A1, nx, nx);
    //cout<<"TRSM: \n"<<tt<<"\n";
    //printf("TRSM:%llx \n", task->tag_id);
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
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nx, nx, ny, 
    -1.0,A0, nx, A1, nx, 1.0, A2, nx);
    //printf("GEMM:%llx \n", task->tag_id);
  }
struct starpu_codelet gemm_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { gemm, NULL },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_RW }
};






void cholesky(int n, int nb, int rank, int size, int test, int nrow, int ncol) {
    auto val = [&](int i, int j) { return  1/(float)((i-j)*(i-j)+1); };
    auto distrib = [&](int i, int j) { return  ((i+j*nb) % size == rank); };
    MatrixXd B=MatrixXd::NullaryExpr(n*nb,n*nb, val);
    MatrixXd L = B;
    vector<MatrixXd*> blocs(nb*nb);
    vector<starpu_data_handle_t> dataA(nb*nb);
    auto block_2_rank = [&](int i, int j){return (i % nrow) * ncol + j % ncol;};

    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {

                blocs[ii+jj*nb]=new MatrixXd(n,n);
                *blocs[ii+jj*nb]=L.block(ii*n,jj*n,n,n);
                int mpi_rank = block_2_rank(ii,jj);
                if (mpi_rank == rank) {
                    starpu_matrix_data_register(&dataA[ii+jj*nb], STARPU_MAIN_RAM, (uintptr_t)blocs[ii+jj*nb]->data(), n, n, n, sizeof(double));
                }
                else {
                    starpu_matrix_data_register(&dataA[ii+jj*nb], -1, (uintptr_t)NULL, n, n, n, sizeof(double));
                }
                if (dataA[ii+jj*nb]) {
                    starpu_mpi_data_register(dataA[ii+jj*nb], ii+jj*nb, mpi_rank);

                }
        }
    }

    double start = starpu_timing_now();
    for (int kk = 0; kk < nb; ++kk) {
            starpu_mpi_task_insert(MPI_COMM_WORLD,&potrf_cl,STARPU_RW, dataA[kk+kk*nb],0);
        for (int ii = kk+1; ii < nb; ++ii) {
            starpu_mpi_task_insert(MPI_COMM_WORLD,&trsm_cl,STARPU_R, dataA[kk+kk*nb],STARPU_RW, dataA[ii+kk*nb],0);
            starpu_mpi_cache_flush(MPI_COMM_WORLD, dataA[kk+kk*nb]);
            for (int jj=kk+1; jj < nb; ++jj) {         
                if (jj <= ii) {
                    if (jj==ii) {
                        starpu_mpi_task_insert(MPI_COMM_WORLD,&syrk_cl, STARPU_R, dataA[ii+kk*nb],STARPU_RW, dataA[ii+jj*nb],0);
                    }
                    else {
                        starpu_mpi_task_insert(MPI_COMM_WORLD,&gemm_cl,STARPU_R, dataA[ii+kk*nb],STARPU_R, dataA[jj+kk*nb],STARPU_RW, dataA[ii+jj*nb],0);
                    }
                }
            }
            starpu_mpi_cache_flush(MPI_COMM_WORLD, dataA[ii+kk*nb]);
        }
    }
    

    starpu_task_wait_for_all();
    double end = starpu_timing_now();
    if (rank==0) {printf("Elapsed time: %0.4f \n", (end-start)/1000000);}

    MPI_Status status;
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            starpu_data_unregister(dataA[ii+jj*nb]); 
        }
    }

   if (test)     
   {
    for (int ii=0; ii<nb; ii++) {
            for (int jj=0; jj<nb; jj++) {
                if (jj<=ii)  {
                int mpi_rank = block_2_rank(ii,jj);
                if (rank==0 && rank!=mpi_rank) {
                    MPI_Recv(blocs[ii+jj*nb]->data(), n*n, MPI_DOUBLE, mpi_rank, mpi_rank, MPI_COMM_WORLD, &status);
                    }

                else if (rank==mpi_rank && rank != 0 ) {
                    MPI_Send(blocs[ii+jj*nb]->data(), n*n, MPI_DOUBLE, 0, mpi_rank, MPI_COMM_WORLD);
                    }
                }
            }
        }


        if (rank==0) {
        for (int ii=0; ii<nb; ii++) {
            for (int jj=0; jj<nb; jj++) {
                L.block(ii*n,jj*n,n,n)=*blocs[ii+jj*nb];
                free(blocs[ii+jj*nb]);
            }
        }
        auto L1=L.triangularView<Lower>();
        VectorXd x = VectorXd::Random(n * nb);
        VectorXd b = B*x;
        VectorXd bref = b;
        L1.solveInPlace(b);
        L1.transpose().solveInPlace(b);
        double error = (b - x).norm() / x.norm();
        cout << "Error solve: " << error << endl;
        cout<<"rank "<<rank<<" finished....\n";
        
        }
    
   }

}





//Test
int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
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
    int n=10;
    int nb=1;
    int test=0;
    int nrow=1;
    int ncol=1;
    if (argc >= 2)
    {
        n = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        nb = atoi(argv[2]);
    }

    if (argc >= 4) {
        test = atoi(argv[3]);
    }

    if (argc >= 6) {
        nrow = atoi(argv[4]);
        ncol = atoi(argv[5]);

    }
    printf("Usage: ./cholesky_mpi n nb test nrow ncol\n");
    cholesky(n,nb, rank, size, test, nrow, ncol);
    //test(rank);
    starpu_mpi_shutdown();
    MPI_Finalize();
    return 0;
}