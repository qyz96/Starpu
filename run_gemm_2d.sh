#!/bin/bash

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname
STARPU_SCHED=dmdas mpirun -n ${SLURM_NTASKS} ./gemm_2d_mpi ${BLOCK_SIZE} ${NUM_BLOCKS} 0 ${NROWS} ${NCOLS}
