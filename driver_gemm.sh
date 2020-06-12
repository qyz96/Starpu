#!/bin/bash

name="starpu_gemm_%j.out"

NROWS=1 NCOLS=1 BLOCK_SIZE=256 NUM_BLOCKS=16  sbatch --output=${name} -c 32 -n 1 run_gemm_2d.sh
NROWS=1 NCOLS=1 BLOCK_SIZE=256 NUM_BLOCKS=32  sbatch --output=${name} -c 32 -n 1 run_gemm_2d.sh
NROWS=1 NCOLS=1 BLOCK_SIZE=256 NUM_BLOCKS=64  sbatch --output=${name} -c 32 -n 1 run_gemm_2d.sh

NROWS=2 NCOLS=4 BLOCK_SIZE=256 NUM_BLOCKS=16  sbatch --output=${name} -c 32 -n 8 run_gemm_2d.sh
NROWS=2 NCOLS=4 BLOCK_SIZE=256 NUM_BLOCKS=32  sbatch --output=${name} -c 32 -n 8 run_gemm_2d.sh
NROWS=2 NCOLS=4 BLOCK_SIZE=256 NUM_BLOCKS=64  sbatch --output=${name} -c 32 -n 8 run_gemm_2d.sh
NROWS=2 NCOLS=4 BLOCK_SIZE=256 NUM_BLOCKS=128 sbatch --output=${name} -c 32 -n 8 run_gemm_2d.sh

NROWS=8 NCOLS=8 BLOCK_SIZE=256 NUM_BLOCKS=16  sbatch --output=${name} -c 32 -n 64 run_gemm_2d.sh
NROWS=8 NCOLS=8 BLOCK_SIZE=256 NUM_BLOCKS=32  sbatch --output=${name} -c 32 -n 64 run_gemm_2d.sh
NROWS=8 NCOLS=8 BLOCK_SIZE=256 NUM_BLOCKS=64  sbatch --output=${name} -c 32 -n 64 run_gemm_2d.sh
NROWS=8 NCOLS=8 BLOCK_SIZE=256 NUM_BLOCKS=128 sbatch --output=${name} -c 32 -n 64 run_gemm_2d.sh