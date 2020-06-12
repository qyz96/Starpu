#!/bin/bash

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname
STARPU_SCHED=${SCHED} mpirun -n ${SLURM_NTASKS} ./cholesky_mpi ${BLOCK_SIZE} ${NUM_BLOCKS} ${TEST} ${NROWS} ${NCOLS} ${PRUNE}
