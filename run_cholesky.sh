#!/bin/bash
#SBATCH -o starpu_chol_%j.out

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname

MKL_NUM_THREADS=1 STARPU_SCHED=${SCHED} mpirun -n ${SLURM_NTASKS} ./cholesky_mpi ${BLOCK_SIZE} ${NUM_BLOCKS} ${TEST} ${NPROWS} ${NPCOLS} ${PRUNE}
