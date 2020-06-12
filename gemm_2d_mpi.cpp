#include "gemm_shared.hpp"

using namespace std;
using namespace Eigen;

void gemm_2d(int block_size, int num_blocks, int rank, int nranks, int test, int nrow, int ncol) {
    
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

    size_t pruned = 0;

    starpu_mpi_barrier(MPI_COMM_WORLD);
    double start = starpu_timing_now();
    for (int ii = 0; ii < num_blocks; ++ii) {
        for (int jj = 0; jj < num_blocks; ++jj) {
            for (int kk = 0; kk < num_blocks; ++kk) {
                if(block_2_rank(ii,kk) == rank || block_2_rank(kk,jj) == rank || block_2_rank(ii,jj) == rank) {
                    starpu_mpi_task_insert(MPI_COMM_WORLD,&gemm_cl,
                        STARPU_R, dataA[ii+kk*num_blocks],
                        STARPU_R, dataB[kk+jj*num_blocks],
                        STARPU_RW,dataC[ii+jj*num_blocks],
                    0);
                } else {
                    pruned++;
                }
            }
        }
    }
    starpu_task_wait_for_all();
    starpu_mpi_barrier(MPI_COMM_WORLD);
    double end = starpu_timing_now();
    
    int matrix_size = block_size * num_blocks;
    printf("pruned=%zd\n", pruned);
    // Makes grep/import to excel easier ; just do
    // cat output | grep -P '\[0\]\>\>\>\>'
    // to extract rank 0 info
    printf(">>>>test,rank,nranks,matrix_size,block_size,num_blocks,total_time\n");
    printf("[%d]>>>>gemm_2d_starpu,%d,%d,%d,%d,%d,%e\n",rank,rank,nranks,matrix_size,block_size,num_blocks,(end-start)/1e6);

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
    assert(nrow * ncol == nranks);
    printf("Usage: ./gemm_2d block_size num_blocks test nrow ncol\n");
    printf("block_size,%d\n",block_size);
    printf("num_blocks,%d\n",num_blocks);
    printf("test,%d\n",test);
    printf("nprocs_row,%d\n",nrow);
    printf("nprocs_col,%d\n",ncol);
    gemm_2d(block_size, num_blocks, rank, nranks, test, nrow, ncol);
    starpu_mpi_shutdown();
    MPI_Finalize();
    return 0;
}
