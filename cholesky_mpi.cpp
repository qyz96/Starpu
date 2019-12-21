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



void task1(void *buffers[], void *cl_arg) { 

    int *A= (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
	*A+=1;
    cout<<"Incrementing *a, *a="<<*A<<endl;
    return;
     }
struct starpu_codelet cl1 = {
    .where = STARPU_CPU,
    .cpu_funcs = { task1, NULL },
    .nbuffers = 1,
    .modes = { STARPU_RW }
};

void task2(void *buffers[], void *cl_arg) { 

    int *A0= (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);
	int *A1= (int *)STARPU_VARIABLE_GET_PTR(buffers[1]);
    *A1+=*A0;
    cout<<"*b + *a= "<<*A1<<endl;
    return;
     }
struct starpu_codelet cl2 = {
    .where = STARPU_CPU,
    .cpu_funcs = { task2, NULL },
    .nbuffers = 2,
    .modes = { STARPU_R, STARPU_RW }
};


void test(int rank)  {
    int* a=new int(1);
    int* b=new int(1);
    int* c=new int(1);
    starpu_data_handle_t data1, data2;
    if (rank==0) {
        starpu_variable_data_register(&data1, STARPU_MAIN_RAM, (uintptr_t)a, sizeof(int));
        starpu_variable_data_register(&data2, -1, (uintptr_t)NULL, sizeof(int));
    }
    else {
        starpu_variable_data_register(&data1, -1, (uintptr_t)NULL, sizeof(int));
        starpu_variable_data_register(&data2, STARPU_MAIN_RAM, (uintptr_t)b, sizeof(int));
    }
    starpu_mpi_data_register(data1, 0, 0);
    starpu_mpi_data_register(data2, 1, 1);

    starpu_mpi_task_insert(MPI_COMM_WORLD,&cl1, STARPU_RW, data1, 0);
    starpu_mpi_task_insert(MPI_COMM_WORLD,&cl2, STARPU_R, data1,STARPU_RW, data2,0);


    return;


}


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
    auto L = A0->triangularView<Lower>().transpose();
    //cout<<"\n"<<*A0<<"\n"<<*A1<<endl;
    //MatrixXd TT = *A1;
    //auto BB = L.solve<OnTheRight>(TT);
    
	//cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, A0->rows(), A0->rows(), 1.0, A0->data(),A0->rows(), A1->data(), A0->rows());
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

static void s_potrf(int k, starpu_data_handle_t data)
{

    struct starpu_task *task;
	task=starpu_mpi_task_build(MPI_COMM_WORLD, &potrf_cl, STARPU_R, data, 0);
	int ret = starpu_task_submit(task);
    starpu_mpi_task_post_build(MPI_COMM_WORLD, &potrf_cl, STARPU_R, data, 0);
}

static void s_trsm(int k, int i, starpu_data_handle_t data1,  starpu_data_handle_t data2)
{
    //printf("task 21 k = %d i = %d TAG = %llx\n", k, i, (TAG21(k, i)));
    struct starpu_task *task;

	task=starpu_mpi_task_build(MPI_COMM_WORLD, &trsm_cl, STARPU_R, data1, STARPU_RW, data2, 0);
	int ret = starpu_task_submit(task);
    starpu_mpi_task_post_build(MPI_COMM_WORLD, &trsm_cl, STARPU_R, data1, STARPU_RW, data2, 0);

}

static void s_gemm(int k, int i, int j, starpu_data_handle_t data1, starpu_data_handle_t data2, starpu_data_handle_t data3)
{
    //printf("task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG22(k,i,j)); 

	struct starpu_task *task;
    if (i==j){
        task=starpu_mpi_task_build(MPI_COMM_WORLD, &syrk_cl, STARPU_R, data1, STARPU_RW, data2, 0);
    }
    else {
        task=starpu_mpi_task_build(MPI_COMM_WORLD, &gemm_cl, STARPU_R, data1, STARPU_R, data2, STARPU_RW, data3,0);
    }
    int ret = starpu_task_submit(task);
    if (i==j){
        starpu_mpi_task_post_build(MPI_COMM_WORLD, &syrk_cl, STARPU_R, data1, STARPU_RW, data2, 0);
    }
    else {
        starpu_mpi_task_post_build(MPI_COMM_WORLD, &gemm_cl, STARPU_R, data1, STARPU_R, data2, STARPU_RW, data3,0);
    }
}


void cholesky(int n, int nb, int rank, int size) {
    auto val = [&](int i, int j) { return  1/(float)((i-j)*(i-j)+1); };
    auto distrib = [&](int i, int j) { return  ((i+j*nb) % size == rank); };
    MatrixXd B=MatrixXd::NullaryExpr(n*nb,n*nb, val);
    MatrixXd L = B;
    vector<MatrixXd*> blocs(nb*nb);
    vector<starpu_data_handle_t> dataA(nb*nb);

    cout<<"rank "<<rank<<" registering data....\n";
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {

                blocs[ii+jj*nb]=new MatrixXd(n,n);
                *blocs[ii+jj*nb]=L.block(ii*n,jj*n,n,n);
                //starpu_variable_data_register(&dataA[ii+jj*nb], -1, (uintptr_t)NULL, sizeof(MatrixXd));
                int mpi_rank = ((ii+jj*nb)%size);
                if (mpi_rank == rank) {
                    starpu_variable_data_register(&dataA[ii+jj*nb], STARPU_MAIN_RAM, (uintptr_t)blocs[ii+jj*nb], sizeof(MatrixXd));
                }
                else {
                    starpu_variable_data_register(&dataA[ii+jj*nb], -1, (uintptr_t)NULL, sizeof(MatrixXd));
                }
                if (dataA[ii+jj*nb]) {
                    starpu_mpi_data_register(dataA[ii+jj*nb], ii+jj*nb, mpi_rank);

                }
        }
    }
    MatrixXd* A=&B;
    //cout<<A->size()<<"\n";
    //Test
    cout<<"rank "<<rank<<" inserting tasks....\n";
    double start = starpu_timing_now();
    for (int kk = 0; kk < nb; ++kk) {
            starpu_mpi_task_insert(MPI_COMM_WORLD,&potrf_cl,STARPU_RW, dataA[kk+kk*nb],0);
        for (int ii = kk+1; ii < nb; ++ii) {
            starpu_mpi_task_insert(MPI_COMM_WORLD,&trsm_cl,STARPU_R, dataA[kk+kk*nb],STARPU_RW, dataA[ii+kk*nb],0);
            starpu_mpi_cache_flush(MPI_COMM_WORLD, dataA[kk+kk*nb]);
            for (int jj=kk+1; jj < nb; ++jj) {         
                if (jj <= ii) {
                    if (jj==ii) {
                        //starpu_mpi_task_insert(MPI_COMM_WORLD,&syrk_cl, STARPU_R, dataA[ii+kk*nb],STARPU_RW, dataA[ii+jj*nb],0);
                    }
                    else {
                        //starpu_mpi_task_insert(MPI_COMM_WORLD,&gemm_cl,STARPU_R, dataA[ii+kk*nb],STARPU_R, dataA[jj+kk*nb],STARPU_RW, dataA[ii+jj*nb],0);
                    }
                }
            }
            starpu_mpi_cache_flush(MPI_COMM_WORLD, dataA[ii+kk*nb]);
        }
    }
    
    /* for (int kk = 0; kk < nb; ++kk) {
        if ((kk+kk*nb)%size == rank) {
            s_potrf(kk, dataA[kk+kk*nb]);
        }
        for (int ii = kk+1; ii < nb; ++ii) {
            if ((ii+kk*nb)%size == rank) {
            s_trsm(kk,ii,dataA[kk+kk*nb],dataA[ii+kk*nb]);
            }
            for (int jj=kk+1; jj < nb; ++jj) {         
                if (jj <= ii) {
                        if ((ii+jj*nb)%size == rank) {
                        s_gemm(kk,ii,jj, dataA[ii+kk*nb],dataA[jj+kk*nb], dataA[ii+jj*nb]);
                    }
                }
            }
        }
    } */
    starpu_task_wait_for_all();
    double end = starpu_timing_now();


    printf("Elapsed time: %0.4f \n", (end-start)/1000000);

    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            if (jj <= ii) {
             if ((ii+jj*nb)%size == rank) {
                //cout<<ii<<" "<<jj<<":\n"<<*blocs[ii+jj*nb]<<endl;
            }
            //cout<<ii<<" "<<jj<<endl;
            //starpu_data_release(dataA[ii+jj*nb]);
            
            }
        }
    }
    LLT<Ref<MatrixXd>> llt(L);
    //cout<<"Ref:\n"<<L<<endl;

/*     for (int ii=0; ii<nb; ii++) {
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
    } */
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
    //test(rank);
    starpu_mpi_shutdown();
    return 0;
}