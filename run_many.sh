#!/bin/bash

hostname
lscpu

BLOCK_SIZE[0]=32
BLOCK_SIZE[1]=64
BLOCK_SIZE[2]=128
BLOCK_SIZE[3]=256
BLOCK_SIZE[4]=512
BLOCK_SIZE[5]=1024
BLOCK_SIZE[6]=2048
    
mpirun -n ${SLURM_NTASKS} hostname

for i in 1 2 3 4 5 6
do

    let NUM_BLOCKS=$((${MATRIX_SIZE}/${BLOCK_SIZE[i]}))

    echo ${MATRIX_SIZE}
    echo ${BLOCK_SIZE[i]}
    echo ${NUM_BLOCKS}
    
    STARPU_SCHED=${SCHED} mpirun -n ${SLURM_NTASKS} ./cholesky_mpi ${BLOCK_SIZE[i]} ${NUM_BLOCKS} ${TEST} ${NROWS} ${NCOLS} ${PRUNE}
    echo "=================================="

done
