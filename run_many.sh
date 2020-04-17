#!/bin/bash
#SBATCH -o starpu_chol.%j.out

hostname
lscpu

BLOCK_SIZE[0]=64
BLOCK_SIZE[1]=128
BLOCK_SIZE[2]=256
BLOCK_SIZE[3]=512
BLOCK_SIZE[4]=1024
    
mpirun -n ${SLURM_NTASKS} hostname

for i in {0..4}
do

    let NUM_BLOCKS=$((${MATRIX_SIZE}/${BLOCK_SIZE[i]}))

    echo ${MATRIX_SIZE}
    echo ${BLOCK_SIZE[i]}
    echo ${NUM_BLOCKS}
    
    STARPU_SCHED=${SCHED} mpirun -n ${SLURM_NTASKS} ./cholesky_mpi ${BLOCK_SIZE[i]} ${NUM_BLOCKS} ${TEST} ${NROWS} ${NCOLS}
    echo "=================================="

done
