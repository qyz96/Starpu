#include "gemm_shared.hpp"

using namespace std;
using namespace Eigen;

/**
 * This hangs and does not work :-(
 */
void gemm_3d(const int block_size, const int num_blocks, const int rank, const int n_ranks, const int test) {
    
    const int n_ranks_1d = static_cast<int>(round(pow(n_ranks, 1.0/3.0)));
    assert(n_ranks_1d * n_ranks_1d * n_ranks_1d == n_ranks);
    assert(num_blocks % n_ranks_1d == 0);
    const int blocks_per_ranks_1d = num_blocks / n_ranks_1d;
    const int rank_i = rank % n_ranks_1d;
    const int rank_j = (rank / n_ranks_1d) % n_ranks_1d;
    const int rank_k = rank / (n_ranks_1d * n_ranks_1d);

    vector<MatrixXd*> blocksA(num_blocks*num_blocks);
    vector<MatrixXd*> blocksB(num_blocks*num_blocks);
    vector<MatrixXd*> blocksC(num_blocks*num_blocks);
    vector<starpu_data_handle_t> dataA(num_blocks*num_blocks);
    vector<starpu_data_handle_t> dataB(num_blocks*num_blocks);
    vector<starpu_data_handle_t> dataC(num_blocks*num_blocks);

    auto block_2_rank = [&](int i, int j){
        return (i / blocks_per_ranks_1d) + (j / blocks_per_ranks_1d) * n_ranks_1d;
    };

    for (int ii=0; ii<num_blocks; ii++) {
        for (int jj=0; jj<num_blocks; jj++) {
            blocksA[ii+jj*num_blocks] = new MatrixXd();
            blocksB[ii+jj*num_blocks] = new MatrixXd();
            blocksC[ii+jj*num_blocks] = new MatrixXd();
            const int mpi_rank = block_2_rank(ii, jj);
            if (rank_k == 0 && mpi_rank == rank) {
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

    printf("Starting computation\n");
    size_t pruned = 0;

    starpu_mpi_barrier(MPI_COMM_WORLD);
    double start = starpu_timing_now();
    for (int ii = 0; ii < num_blocks; ++ii) {
        // printf("ii = %d\n", ii);
        for (int jj = 0; jj < num_blocks; ++jj) {
            for (int kk = 0; kk < num_blocks; ++kk) {
                const int compute_at = (ii / blocks_per_ranks_1d) + (jj / blocks_per_ranks_1d) * n_ranks_1d + (kk / blocks_per_ranks_1d) * n_ranks_1d * n_ranks_1d;
                // printf("compute_at %d\n", compute_at);
                // if(
                //     (rank_k == 0 && block_2_rank(ii, kk) == rank) ||
                //     (rank_k == 0 && block_2_rank(kk, jj) == rank) ||
                //     (rank_k == 0 && block_2_rank(ii, jj) == rank) ||
                //     (rank == compute_at)
                // ) {
                    starpu_mpi_task_insert(MPI_COMM_WORLD,&gemm_cl,
                        STARPU_R,               dataA[ii+kk*num_blocks], // Eventually located at rank_k = 0
                        STARPU_R,               dataB[kk+jj*num_blocks],
                        STARPU_RW,              dataC[ii+jj*num_blocks],
                        STARPU_EXECUTE_ON_NODE, compute_at,
                        0);
                // } else {
                //     pruned++;
                // }
            }
        }
    }
    printf("(%d)\n", rank);
    starpu_task_wait_for_all();
    starpu_mpi_barrier(MPI_COMM_WORLD);
    double end = starpu_timing_now();

    int matrix_size = block_size * num_blocks;
    printf("pruned=%zd\n", pruned);
    // Makes grep/import to excel easier ; just do
    // cat output | grep -P '\[0\]\>\>\>\>'
    // to extract rank 0 info
    printf(">>>>test,rank,n_ranks,matrix_size,block_size,num_blocks,total_time\n");
    printf("[%d]>>>>gemm_3d_starpu,%d,%d,%d,%d,%d,%e\n",rank,rank,n_ranks,matrix_size,block_size,num_blocks,(end-start)/1e6);

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
                const int mpi_rank = block_2_rank(ii, jj);
                if (rank_k == 0 && rank == 0 && mpi_rank != rank) {
                    blocksC[ii+jj*num_blocks] = new MatrixXd(block_size, block_size);
                    MPI_Recv(blocksC[ii+jj*num_blocks]->data(), block_size*block_size, MPI_DOUBLE, mpi_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (rank_k == 0 && rank != 0 && mpi_rank == rank) {
                    MPI_Send(blocksC[ii+jj*num_blocks]->data(), block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }
        }
        if (rank == 0) {
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
    int rank, n_ranks;
    starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
    starpu_mpi_comm_size(MPI_COMM_WORLD, &n_ranks);
    if (rank==0) {
        printf("Running on %d CPU cores per rank,", starpu_worker_get_count_by_type(STARPU_CPU_WORKER));
        printf("and %d ranks in total\n", n_ranks);
    }
    int block_size=10;
    int num_blocks=1;
    int test=0;
    if (argc >= 2) {
        block_size = atoi(argv[1]);
    }

    if (argc >= 3) {
        num_blocks = atoi(argv[2]);
    }

    if (argc >= 4) {
        test = atoi(argv[3]);
    }

    printf("Usage: ./gemm_3d block_size num_blocks test\n");
    printf("block_size,%d\n",block_size);
    printf("num_blocks,%d\n",num_blocks);
    printf("test,%d\n",test);
    gemm_3d(block_size, num_blocks, rank, n_ranks, test);
    starpu_mpi_shutdown();
    MPI_Finalize();
    return 0;
}
