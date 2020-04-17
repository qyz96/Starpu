#!/bin/bash

SCHED=dmdas NROWS=1 NCOLS=1 MATRIX_SIZE=4096  TEST=0 sbatch -c 32 -n 1 run_many.sh
SCHED=dmdas NROWS=1 NCOLS=1 MATRIX_SIZE=8192  TEST=0 sbatch -c 32 -n 1 run_many.sh
SCHED=dmdas NROWS=1 NCOLS=1 MATRIX_SIZE=16384 TEST=0 sbatch -c 32 -n 1 run_many.sh

SCHED=dmdas NROWS=2 NCOLS=4 MATRIX_SIZE=4096  TEST=0 sbatch -c 32 -n 8 run_many.sh
SCHED=dmdas NROWS=2 NCOLS=4 MATRIX_SIZE=8192  TEST=0 sbatch -c 32 -n 8 run_many.sh
SCHED=dmdas NROWS=2 NCOLS=4 MATRIX_SIZE=16384 TEST=0 sbatch -c 32 -n 8 run_many.sh
SCHED=dmdas NROWS=2 NCOLS=4 MATRIX_SIZE=32768 TEST=0 sbatch -c 32 -n 8 run_many.sh

SCHED=dmdas NROWS=8 NCOLS=8 MATRIX_SIZE=4096  TEST=0 sbatch -c 32 -n 64 run_many.sh
SCHED=dmdas NROWS=8 NCOLS=8 MATRIX_SIZE=8192  TEST=0 sbatch -c 32 -n 64 run_many.sh
SCHED=dmdas NROWS=8 NCOLS=8 MATRIX_SIZE=16384 TEST=0 sbatch -c 32 -n 64 run_many.sh
SCHED=dmdas NROWS=8 NCOLS=8 MATRIX_SIZE=32768 TEST=0 sbatch -c 32 -n 64 run_many.sh
SCHED=dmdas NROWS=8 NCOLS=8 MATRIX_SIZE=65536 TEST=0 sbatch -c 32 -n 64 run_many.sh
