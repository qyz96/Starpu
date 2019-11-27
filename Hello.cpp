#include <starpu.h>
#include<iostream>
#include <cblas.h>
#include <lapacke.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>

using namespace std;
using namespace Eigen;



void cpu_func(void *buffers[], void *cl_arg)
{
    printf("Hello world\n");
}
struct starpu_codelet cl =
{
    .cpu_funcs = { cpu_func },
    .nbuffers = 0
};

void test(void *buffers[], void *cl_arg)
{
	// get the current task
	auto task = starpu_task_get_current();
    //int* ss=n;
	// get the user data (pointers to the vec_A, vec_B, vec_C std::vector)
	auto u_data0 = starpu_data_get_user_data(task->handles[0]); assert(u_data0);

	// cast void* in std::vector<char>*
	auto A = static_cast<MatrixXd*>(u_data0);


	// all the std::vector have to have the same size
	cout<<A->size()<<"\n";
}

//Test
int main()
{
    int n=4;
    int nb=4;
    auto val = [&](int i, int j) { return 1/(float)((i-j)*(i-j)+1); };
    MatrixXd A;
    A = MatrixXd::NullaryExpr(nb,nb, val);
    cout<<A.size()<<"\n";
    
    return 0;
}