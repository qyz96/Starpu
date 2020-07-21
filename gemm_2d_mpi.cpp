#include "gemm_shared.hpp"

using namespace std;
using namespace Eigen;

void gemm_2d(const int block_size, const int num_blocks, const int rank, const int nranks, const int test, const int nrow, const int ncol, const bool prune) {
    
    // Warmup MKL
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(256,256);
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(256,256);
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(256,256);
    for(int i = 0; i < 10; i++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 256, 256, 256, 1.0, A.data(), 256, B.data(), 256, 1.0, C.data(), 256);
    }
    
    vector<MatrixXd*> blocksA(num_blocks*num_blocks);
    vector<MatrixXd*> blocksB(num_blocks*num_blocks);
    vector<MatrixXd*> blocksC(num_blocks*num_blocks);
    vector<starpu_data_handle_t> dataA(num_blocks*num_blocks);
    vector<starpu_data_handle_t> dataB(num_blocks*num_blocks);
    vector<starpu_data_handle_t> dataC(num_blocks*num_blocks);
    auto block_2_rank = [&](int i, int j){return (i % nrow) * ncol + j % ncol;};

    for (int ii=0; ii<num_blocks; ii++) {
        for (int jj=0; jj<num_blocks; jj++) {
            int mpi_rank = block_2_rank(ii,jj);
            blocksA[ii+jj*num_blocks] = new MatrixXd();
            blocksB[ii+jj*num_blocks] = new MatrixXd();
            blocksC[ii+jj*num_blocks] = new MatrixXd();
            if (mpi_rank == rank) {
                auto val_block = [&](int i, int j) { return val(ii*block_size+i,jj*block_size+j); };
                *blocksA[ii+jj*num_blocks] = MatrixXd::NullaryExpr(block_size, block_size, val_block);
                *blocksB[ii+jj*num_blocks] = MatrixXd::NullaryExpr(block_size, block_size, val_block);
                *blocksC[ii+jj*num_blocks] = MatrixXd::Zero(block_size, block_size);
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
                starpu_matrix_data_register(&dataA[ii+jj*num_blocks], STARPU_MAIN_RAM, (uintptr_t)blocksA[ii+jj*num_blocks]->data(), block_size, block_size, block_size, sizeof(double));
                starpu_matrix_data_register(&dataB[ii+jj*num_blocks], STARPU_MAIN_RAM, (uintptr_t)blocksB[ii+jj*num_blocks]->data(), block_size, block_size, block_size, sizeof(double));
                starpu_matrix_data_register(&dataC[ii+jj*num_blocks], STARPU_MAIN_RAM, (uintptr_t)blocksC[ii+jj*num_blocks]->data(), block_size, block_size, block_size, sizeof(double));
            } else {
                starpu_matrix_data_register(&dataA[ii+jj*num_blocks], -1, (uintptr_t)NULL, block_size, block_size, block_size, sizeof(double));
                starpu_matrix_data_register(&dataB[ii+jj*num_blocks], -1, (uintptr_t)NULL, block_size, block_size, block_size, sizeof(double));
                starpu_matrix_data_register(&dataC[ii+jj*num_blocks], -1, (uintptr_t)NULL, block_size, block_size, block_size, sizeof(double));
            }
            if (dataA[ii+jj*num_blocks]) starpu_mpi_data_register(dataA[ii+jj*num_blocks], 0                       + ii+jj*num_blocks, mpi_rank);
            if (dataB[ii+jj*num_blocks]) starpu_mpi_data_register(dataB[ii+jj*num_blocks], num_blocks*num_blocks   + ii+jj*num_blocks, mpi_rank);
            if (dataC[ii+jj*num_blocks]) starpu_mpi_data_register(dataC[ii+jj*num_blocks], num_blocks*num_blocks*2 + ii+jj*num_blocks, mpi_rank);
        }
    }

    for (int kk = 0; kk < num_blocks; ++kk) {
    	for (int ii = 0; ii < num_blocks; ++ii) {
            for (int jj = 0; jj < num_blocks; ++jj) {
                if( (!prune) || block_2_rank(ii,kk) == rank || block_2_rank(kk,jj) == rank || block_2_rank(ii,jj) == rank) {
                    starpu_mpi_task_insert(MPI_COMM_WORLD,&gemm_cl,
                        STARPU_R, dataA[ii+kk*num_blocks],
                        STARPU_R, dataB[kk+jj*num_blocks],
                        STARPU_RW,dataC[ii+jj*num_blocks],
                    0);
                } else {
                    num_pruned++;
                }
            }
        }
    }
    
    starpu_task_wait_for_all();
    starpu_mpi_barrier(MPI_COMM_WORLD);
    double end = starpu_timing_now();
    
    int matrix_size = block_size * num_blocks;
    // Makes grep/import to excel easier ; just do
    // cat output | grep -P '\[0\]\>\>\>\>'
    // to extract rank 0 info
    const int ncores = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
    long long int flops_per_rank = ((long long int)matrix_size) * ((long long int)matrix_size) * ((long long int)matrix_size) / ((long long int)nranks);
    long long int flops_per_core = flops_per_rank / ((long long int)ncores);
    printf(">>>>test rank nranks n_cores matrix_size block_size num_blocks total_time flops_per_rank flops_per_core num_pruned\n");
    printf("[%d]>>>>gemm_2d_starpu %d %d %d %d %d %d %e %llu %llu %zd\n",rank,rank,nranks,ncores,matrix_size,block_size,num_blocks,(end-start)/1e6,flops_per_rank,flops_per_core,num_pruned);

    for (int ii=0; ii<num_blocks; ii++) {
        for (int jj=0; jj<num_blocks; jj++) {
            starpu_data_unregister(dataA[ii+jj*num_blocks]); 
            starpu_data_unregister(dataB[ii+jj*num_blocks]); 
            starpu_data_unregister(dataC[ii+jj*num_blocks]); 
        }
    }

    if (test) {
        printf("Testing...\n");
        for (int ii=0; ii<num_blocks; ii++) {
            for (int jj=0; jj<num_blocks; jj++) {
                int mpi_rank = block_2_rank(ii,jj);
                if (rank == 0 && rank != mpi_rank) {
                    blocksC[ii+jj*num_blocks] = new MatrixXd(block_size, block_size);
                    MPI_Recv(blocksC[ii+jj*num_blocks]->data(), block_size*block_size, MPI_DOUBLE, mpi_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (rank == mpi_rank && rank != 0) {
                    MPI_Send(blocksC[ii+jj*num_blocks]->data(), block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }
        }
        if (rank==0) {
            MatrixXd A    = MatrixXd::NullaryExpr(block_size*num_blocks,block_size*num_blocks, val);
            MatrixXd B    = MatrixXd::NullaryExpr(block_size*num_blocks,block_size*num_blocks, val);
            MatrixXd Cref = MatrixXd::NullaryExpr(block_size*num_blocks,block_size*num_blocks, val);
            for (int ii=0; ii<num_blocks; ii++) {
                for (int jj=0; jj<num_blocks; jj++) {
                    Cref.block(ii*block_size,jj*block_size,block_size,block_size) = *blocksC[ii+jj*num_blocks];
                }
            }
            MatrixXd C = A * B;
            double error = (C - Cref).cwiseAbs().maxCoeff() / Cref.cwiseAbs().maxCoeff();
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
    int rank, nranks;
    starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
    starpu_mpi_comm_size(MPI_COMM_WORLD, &nranks);
    if (rank==0) {
        printf("Running on %d CPU cores per rank,", starpu_worker_get_count_by_type(STARPU_CPU_WORKER));
        printf("and %d ranks in total\n", nranks);
    }
    int block_size=10;
    int num_blocks=1;
    int test=0;
    int nrow=1;
    int ncol=1;
    bool prune=true;

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

    assert(nrow * ncol == nranks);
    printf("Usage: ./gemm_2d block_size num_blocks test nrow ncol prune\n");
    printf("block_size,%d\n",block_size);
    printf("num_blocks,%d\n",num_blocks);
    printf("test,%d\n",test);
    printf("nprocs_row,%d\n",nrow);
    printf("nprocs_col,%d\n",ncol);
    printf("prune,%d\n",prune);
    gemm_2d(block_size, num_blocks, rank, nranks, test, nrow, ncol, prune);
    starpu_mpi_shutdown();
    MPI_Finalize();
    return 0;
}
