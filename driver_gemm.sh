#!/bin/bash

NROWS=1 NCOLS=1 BLOCK_SIZE=256 NUM_BLOCKS=32  sbatch -c 32 -n 1 run_gemm_2d.sh
NROWS=1 NCOLS=1 BLOCK_SIZE=256 NUM_BLOCKS=64  sbatch -c 32 -n 1 run_gemm_2d.sh

NROWS=4 NCOLS=2 BLOCK_SIZE=256 NUM_BLOCKS=32  sbatch -c 32 -n 8 run_gemm_2d.sh
NROWS=4 NCOLS=2 BLOCK_SIZE=256 NUM_BLOCKS=64  sbatch -c 32 -n 8 run_gemm_2d.sh
NROWS=4 NCOLS=2 BLOCK_SIZE=256 NUM_BLOCKS=128 sbatch -c 32 -n 8 run_gemm_2d.sh

NROWS=8 NCOLS=8 BLOCK_SIZE=256 NUM_BLOCKS=32  sbatch -c 32 -n 64 run_gemm_2d.sh
NROWS=8 NCOLS=8 BLOCK_SIZE=256 NUM_BLOCKS=64  sbatch -c 32 -n 64 run_gemm_2d.sh
NROWS=8 NCOLS=8 BLOCK_SIZE=256 NUM_BLOCKS=128 sbatch -c 32 -n 64 run_gemm_2d.sh
NROWS=8 NCOLS=8 BLOCK_SIZE=256 NUM_BLOCKS=256 sbatch -c 32 -n 64 run_gemm_2d.sh
