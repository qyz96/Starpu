#!/bin/bash

        
SCHED=dmdas NROWS=1 NCOLS=1 TEST=0 MATRIX_SIZE=4096  sbatch --nodelist=compute-1-4 -c 32 -n 1 run_many.sh
SCHED=dmdas NROWS=1 NCOLS=1 TEST=0 MATRIX_SIZE=8192  sbatch --nodelist=compute-1-5 -c 32 -n 1 run_many.sh
SCHED=dmdas NROWS=1 NCOLS=1 TEST=0 MATRIX_SIZE=16384 sbatch --nodelist=compute-1-6 -c 32 -n 1 run_many.sh

SCHED=dmdas NROWS=1 NCOLS=2 TEST=0 MATRIX_SIZE=4096  sbatch --nodelist=compute-1-[30-31] -c 32 -n 2 run_many.sh
SCHED=dmdas NROWS=1 NCOLS=2 TEST=0 MATRIX_SIZE=8192  sbatch --nodelist=compute-1-[30-31] -c 32 -n 2 run_many.sh
SCHED=dmdas NROWS=1 NCOLS=2 TEST=0 MATRIX_SIZE=16384 sbatch --nodelist=compute-1-[30-31] -c 32 -n 2 run_many.sh

SCHED=dmdas NROWS=2 NCOLS=2 TEST=0 MATRIX_SIZE=4096  sbatch --nodelist=compute-1-[30-33] -c 32 -n 4 run_many.sh
SCHED=dmdas NROWS=2 NCOLS=2 TEST=0 MATRIX_SIZE=8192  sbatch --nodelist=compute-1-[30-33] -c 32 -n 4 run_many.sh
SCHED=dmdas NROWS=2 NCOLS=2 TEST=0 MATRIX_SIZE=16384 sbatch --nodelist=compute-1-[30-33] -c 32 -n 4 run_many.sh

SCHED=dmdas NROWS=2 NCOLS=4 TEST=0 MATRIX_SIZE=4096  sbatch --nodelist=compute-1-[17-24] -c 32 -n 8 run_many.sh
SCHED=dmdas NROWS=2 NCOLS=4 TEST=0 MATRIX_SIZE=8192  sbatch --nodelist=compute-1-[17-24] -c 32 -n 8 run_many.sh
SCHED=dmdas NROWS=2 NCOLS=4 TEST=0 MATRIX_SIZE=16384 sbatch --nodelist=compute-1-[17-24] -c 32 -n 8 run_many.sh
SCHED=dmdas NROWS=2 NCOLS=4 TEST=0 MATRIX_SIZE=32768 sbatch --nodelist=compute-1-[17-24] -c 32 -n 8 run_many.sh

SCHED=dmdas NROWS=4 NCOLS=4 TEST=0 MATRIX_SIZE=4096  sbatch --nodelist=compute-1-[1-16] -c 32 -n 16 run_many.sh
SCHED=dmdas NROWS=4 NCOLS=4 TEST=0 MATRIX_SIZE=8192  sbatch --nodelist=compute-1-[1-16] -c 32 -n 16 run_many.sh
SCHED=dmdas NROWS=4 NCOLS=4 TEST=0 MATRIX_SIZE=16384 sbatch --nodelist=compute-1-[1-16] -c 32 -n 16 run_many.sh
SCHED=dmdas NROWS=4 NCOLS=4 TEST=0 MATRIX_SIZE=32768 sbatch --nodelist=compute-1-[1-16] -c 32 -n 16 run_many.sh

SCHED=dmdas NROWS=4 NCOLS=8 TEST=0 MATRIX_SIZE=4096  sbatch --nodelist=compute-1-[1-32] -c 32 -n 32 run_many.sh
SCHED=dmdas NROWS=4 NCOLS=8 TEST=0 MATRIX_SIZE=8192  sbatch --nodelist=compute-1-[1-32] -c 32 -n 32 run_many.sh
SCHED=dmdas NROWS=4 NCOLS=8 TEST=0 MATRIX_SIZE=16384 sbatch --nodelist=compute-1-[1-32] -c 32 -n 32 run_many.sh
SCHED=dmdas NROWS=4 NCOLS=8 TEST=0 MATRIX_SIZE=32768 sbatch --nodelist=compute-1-[1-32] -c 32 -n 32 run_many.sh

SCHED=dmdas NROWS=8 NCOLS=8 TEST=0 MATRIX_SIZE=4096  sbatch --nodelist=compute-1-[1-34],compute-3-[1-30] -c 32 -n 64 run_many.sh
SCHED=dmdas NROWS=8 NCOLS=8 TEST=0 MATRIX_SIZE=8192  sbatch --nodelist=compute-1-[1-34],compute-3-[1-30] -c 32 -n 64 run_many.sh
SCHED=dmdas NROWS=8 NCOLS=8 TEST=0 MATRIX_SIZE=16384 sbatch --nodelist=compute-1-[1-34],compute-3-[1-30] -c 32 -n 64 run_many.sh
SCHED=dmdas NROWS=8 NCOLS=8 TEST=0 MATRIX_SIZE=32768 sbatch --nodelist=compute-1-[1-34],compute-3-[1-30] -c 32 -n 64 run_many.sh
SCHED=dmdas NROWS=8 NCOLS=8 TEST=0 MATRIX_SIZE=65536 sbatch --nodelist=compute-1-[1-34],compute-3-[1-30] -c 32 -n 64 run_many.sh
