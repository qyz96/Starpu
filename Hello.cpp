#include <starpu.h>
#include<vector>
#include<memory>
#include<iostream>
#include <cblas.h>
#include <lapacke.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>

using namespace std;
using namespace Eigen;


void potrf(void *buffers[], void *cl_arg) { 
	auto task = starpu_task_get_current();
	auto u_data0 = starpu_data_get_user_data(task->handles[0]); 
    double *val = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
	auto A = static_cast<MatrixXd*>(u_data0);
	LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A->rows(), A->data(), A->rows());
    printf("Potrf %d\n", val[0]);
     }
struct starpu_codelet potrf_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { potrf, NULL },
    .nbuffers = 1,
    .modes = { STARPU_RW }
};

void trsm(void *buffers[], void *cl_arg) {
    auto task = starpu_task_get_current();
	auto u_data0 = starpu_data_get_user_data(task->handles[0]); 
	auto A0 = static_cast<MatrixXd*>(u_data0);
    auto u_data1 = starpu_data_get_user_data(task->handles[1]); 
	auto A1 = static_cast<MatrixXd*>(u_data1);
	cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, A0->rows(), 
    A0->rows(), 1.0, A0->data(),A0->rows(), A1->data(), A0->rows());
  }
struct starpu_codelet trsm_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { trsm, NULL },
    .nbuffers = 2,
    .modes = { STARPU_R, STARPU_RW }
};

void syrk(void *buffers[], void *cl_arg) {  }
struct starpu_codelet syrk_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { syrk, NULL },
    .nbuffers = 2,
    .modes = { STARPU_R, STARPU_RW }
};

void gemm(void *buffers[], void *cl_arg) {
    auto task = starpu_task_get_current();
	auto u_data0 = starpu_data_get_user_data(task->handles[0]); 
	auto A0 = static_cast<MatrixXd*>(u_data0);
    auto u_data1 = starpu_data_get_user_data(task->handles[1]); 
	auto A1 = static_cast<MatrixXd*>(u_data1);
    auto u_data2 = starpu_data_get_user_data(task->handles[2]); 
	auto A2 = static_cast<MatrixXd*>(u_data2);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A0->rows(), A0->rows(), A0->rows(), 
    -1.0,A0->data(), A0->rows(), A1->transpose().data(), A0->rows(), 1.0, 
    A2->data(), A0->rows());
  }
struct starpu_codelet gemm_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { gemm, NULL },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_RW }
};
void cpu_func(void *buffers[], void *cl_arg)
{
    printf("Hello world\n");
}




//Test
int main(int argc, char **argv)
{
    int n=4;
    int nb=4;
    if (argc >= 2)
    {
        n = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        nb = atoi(argv[2]);
    }
    auto val = [&](int i, int j) { return 1/(float)((i-j)*(i-j)+1); };
    MatrixXd B=MatrixXd::NullaryExpr(n*nb,n*nb, val);
    MatrixXd L = B;
    vector<MatrixXd*> blocs(nb*nb);
    vector<starpu_data_handle_t> dataA(nb*nb);
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            double ij[2];
            ij[0]=ii;
            ij[1]=jj;
            blocs[ii+jj*nb]=new MatrixXd(n,n);
            *blocs[ii+jj*nb]=L.block(ii*n,jj*n,n,n);
            starpu_vector_data_register(&dataA[ii+jj*nb], STARPU_MAIN_RAM, (uintptr_t)ij, 
            2, sizeof(double));
            starpu_data_set_user_data(dataA[ii+jj*nb], (void*)blocs[ii+jj*nb]);
        }
    }
    MatrixXd* A=&B;
    //cout<<A->size()<<"\n";
 
    starpu_init(NULL);
    for (int kk = 0; kk < nb; ++kk) {
        starpu_insert_task(&potrf_cl,
                           STARPU_RW, dataA[kk+kk*nb],
                           0);

        for (int ii = kk+1; ii < nb; ++ii) {
            starpu_insert_task(&trsm_cl,
                               STARPU_R, dataA[kk+kk*nb],
                               STARPU_RW, dataA[ii+kk*nb],
                               0);
        }

        for (int ii=kk+1; ii < nb; ++ii) {
            for (int jj=kk+1; jj < ii; ++jj) {
                starpu_insert_task(&gemm_cl,
                                   STARPU_R, dataA[ii+kk*nb],
                                   STARPU_R, dataA[jj+kk*nb],
                                   STARPU_RW, dataA[ii+jj*nb],
                                   0);
            }
        }
    }

    starpu_shutdown();
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
    return 0;
}