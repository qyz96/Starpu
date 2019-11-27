#include <starpu.h>
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

	// get the user data (pointers to the vec_A, vec_B, vec_C std::vector)
	auto u_data0 = starpu_data_get_user_data(task->handles[0]); assert(u_data0);

	// cast void* in std::vector<char>*
	auto A = static_cast<MatrixXd*>(u_data0);


	// all the std::vector have to have the same size
	cout<<*A<<"\n";
}

//Test
int main()
{
    int n=10;
    int nb=10;
    auto val = [&](int i, int j) { return 1/(float)((i-j)*(i-j)+1); };
    MatrixXd *A;
    *A = MatrixXd::NullaryExpr(n*nb,n*nb, val);
    starpu_data_handle_t spu_T;
    starpu_vector_data_register(&spu_T, STARPU_MAIN_RAM, (uintptr_t)A->data(), n, sizeof(double));
    starpu_data_set_user_data(spu_vec_A, (void*)A);
    starpu_codelet cl;
	starpu_codelet_init(&cl);
	cl.cpu_funcs     [0] = test;
	cl.cpu_funcs_name[0] = "test";
	cl.nbuffers          = 1;
	cl.modes         [0] = STARPU_RW;
	cl.name              = "potrf";
    /* initialize StarPU */
    starpu_init(NULL);
    ret = starpu_task_insert(&cl, STARPU_RW, spu_T, 0);

    /* terminate StarPU */
    starpu_shutdown();
    return 0;
}