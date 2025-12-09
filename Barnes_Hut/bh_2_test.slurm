#!/bin/bash
#SBATCH --job-name=bh2_test
#SBATCH --output=bh2_%j.out
#SBATCH --error=bh2_%j.err
#SBATCH --time=00:30:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2

# 测试参数
N_BODIES=10000

module load StdEnv/2023
module load gcc/14.3
module load openmpi/5.0.8

echo "=========================================="
echo "Compiling bh_2.cpp..."
echo "=========================================="

mpic++ -std=c++14 -O3 -march=native -fopenmp -lm -o bh_2 bh_2.cpp

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation OK."
echo ""

# 设置 OpenMP 线程数
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "=========================================="
echo "Running bh_2 simulation"
echo "Number of bodies: $N_BODIES"
echo "MPI processes: $SLURM_NTASKS"
echo "OpenMP threads per process: $OMP_NUM_THREADS"
echo "Total threads: $((SLURM_NTASKS * OMP_NUM_THREADS))"
echo "Start time: $(date)"
echo "=========================================="

# 运行程序
# 
mpirun -np $SLURM_NTASKS ./bh_2 $N_BODIES

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="

