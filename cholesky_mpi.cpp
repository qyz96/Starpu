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

    MatrixXd *A= (MatrixXd *)STARPU_VARIABLE_GET_PTR(buffers[0]);
	LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A->rows(), A->data(), A->rows());
     }
struct starpu_codelet potrf_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { potrf, NULL },
    .nbuffers = 1,
    .modes = { STARPU_RW }
};

void trsm(void *buffers[], void *cl_arg) {
	MatrixXd *A0= (MatrixXd *)STARPU_VARIABLE_GET_PTR(buffers[0]);
	MatrixXd *A1= (MatrixXd *)STARPU_VARIABLE_GET_PTR(buffers[1]);
	cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, A0->rows(), 
    A0->rows(), 1.0, A0->data(),A0->rows(), A1->data(), A0->rows());
    //printf("TRSM:%llx \n", task->tag_id);
  }
struct starpu_codelet trsm_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { trsm, NULL },
    .nbuffers = 2,
    .modes = { STARPU_R, STARPU_RW }
};

void syrk(void *buffers[], void *cl_arg) { 
    MatrixXd *A0= (MatrixXd *)STARPU_VARIABLE_GET_PTR(buffers[0]);
	MatrixXd *A1= (MatrixXd *)STARPU_VARIABLE_GET_PTR(buffers[1]);
	cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, A0->rows(), A0->rows(), -1.0, A0->data(), A0->rows(), 1.0, A1->data(), A0->rows());
 }
struct starpu_codelet syrk_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { syrk, NULL },
    .nbuffers = 2,
    .modes = { STARPU_R, STARPU_RW }
};

void gemm(void *buffers[], void *cl_arg) {
    MatrixXd *A0= (MatrixXd *)STARPU_VARIABLE_GET_PTR(buffers[0]);
	MatrixXd *A1= (MatrixXd *)STARPU_VARIABLE_GET_PTR(buffers[1]);
    MatrixXd *A2= (MatrixXd *)STARPU_VARIABLE_GET_PTR(buffers[2]);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A0->rows(), A0->rows(), A0->rows(), 
    -1.0,A0->data(), A0->rows(), A1->transpose().data(), A0->rows(), 1.0, 
    A2->data(), A0->rows());
    //printf("GEMM:%llx \n", task->tag_id);
  }
struct starpu_codelet gemm_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { gemm, NULL },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_RW }
};





void cholesky(int n, int nb, int rank, int size) {
    auto val = [&](int i, int j) { return  1/(float)((i-j)*(i-j)+1); };
    MatrixXd B=MatrixXd::NullaryExpr(n*nb,n*nb, val);
    MatrixXd L = B;
    vector<MatrixXd*> blocs(nb*nb);
    vector<starpu_data_handle_t> dataA(nb*nb);
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            blocs[ii+jj*nb]=new MatrixXd(n,n);
            *blocs[ii+jj*nb]=L.block(ii*n,jj*n,n,n);
            //starpu_variable_data_register(&dataA[ii+jj*nb], -1, (uintptr_t)NULL, sizeof(MatrixXd));
            starpu_variable_data_register(&dataA[ii+jj*nb], STARPU_MAIN_RAM, (uintptr_t)blocs[ii+jj*nb], sizeof(MatrixXd));
            starpu_mpi_data_register(dataA[ii+jj*nb], ii+jj*nb, rank);
        }
    }
    MatrixXd* A=&B;
    //cout<<A->size()<<"\n";
    //Test
    //starpu_init(NULL);

    double start = starpu_timing_now();
    for (int kk = 0; kk < nb; ++kk) {
        if ((kk+kk*nb)%size == rank) {
            starpu_mpi_task_insert(MPI_COMM_WORLD,&potrf_cl,STARPU_RW, dataA[kk+kk*nb],STARPU_TAG_ONLY, TAG11(kk),0);
        }
        for (int ii = kk+1; ii < nb; ++ii) {
            if ((ii+kk*nb)%size == rank) {
            starpu_mpi_task_insert(MPI_COMM_WORLD,&trsm_cl,STARPU_R, dataA[kk+kk*nb],STARPU_RW, dataA[ii+kk*nb],STARPU_TAG_ONLY, TAG21(kk,ii),0);
            }
            for (int jj=kk+1; jj < nb; ++jj) {         
                if (jj <= ii) {
                    if (jj==ii) {
                        if ((ii+jj*nb)%size == rank) {
                        starpu_mpi_task_insert(MPI_COMM_WORLD,&syrk_cl, STARPU_R, dataA[ii+kk*nb],STARPU_RW, dataA[ii+jj*nb],STARPU_TAG_ONLY, TAG22(kk,ii,jj),0);
                        }
                    }
                    else {
                        if ((ii+jj*nb)%size == rank) {
                        starpu_mpi_task_insert(MPI_COMM_WORLD,&gemm_cl,STARPU_R, dataA[ii+kk*nb],STARPU_R, dataA[jj+kk*nb],STARPU_RW, dataA[ii+jj*nb],STARPU_TAG_ONLY, TAG22(kk,ii,jj),0);
                        }
                    }
                }
            }
        }
    }
    starpu_task_wait_for_all();
    double end = starpu_timing_now();


    printf("Elapsed time: %0.4f \n", (end-start)/1000000);

    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            if (jj <= ii) {
            if (rank==0 && (ii+jj*nb)%size != rank) {
                starpu_data_acquire(dataA[ii+jj*nb], STARPU_W);
                starpu_mpi_irecv_detached(dataA[ii+jj*nb], (ii+jj*nb)%size, ii+jj*nb, MPI_COMM_WORLD, NULL, NULL);
            }
            else if ((ii+jj*nb)%size == rank && rank != 0) {
                starpu_data_acquire(dataA[ii+jj*nb], STARPU_R);
                starpu_mpi_isend_detached(dataA[ii+jj*nb], 0, ii+jj*nb, MPI_COMM_WORLD, NULL, NULL);
            }
            //cout<<ii<<" "<<jj<<endl;
            //starpu_data_release(dataA[ii+jj*nb]);
            
            }
        }
    }
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            L.block(ii*n,jj*n,n,n)=*blocs[ii+jj*nb];
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
}



//Test
int main(int argc, char **argv)
{
    starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
    int rank, size;
    starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
    starpu_mpi_comm_size(MPI_COMM_WORLD, &size);
    printf("Rank %d of %d ranks\n", rank, size);
    int n=10;
    int nb=1;
    if (argc >= 2)
    {
        n = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        nb = atoi(argv[2]);
    }

    cholesky(n,nb, rank, size);
    starpu_mpi_shutdown();
    return 0;
}