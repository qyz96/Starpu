#!/bin/bash
#
#SBATCH --output=starpu_gemm_2d_%j.out

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname
MKL_NUM_THREADS=1 STARPU_SCHED=dmdas mpirun -n ${SLURM_NTASKS} ./gemm_2d_mpi ${BLOCK_SIZE} ${NUM_BLOCKS} 0 ${NROWS} ${NCOLS}
