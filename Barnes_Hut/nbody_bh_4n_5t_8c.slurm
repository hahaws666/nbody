#!/bin/bash
#SBATCH --job-name=nbody_4n_5t
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=nbody_4n_5t_8c_%j.out
#SBATCH --error=nbody_4n_5t_8c_%j.err

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Total Tasks: $SLURM_NTASKS"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Total CPUs: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "=========================================="

# 加载必要的模块
module load StdEnv/2023
module load gcc/14.3
module load openmpi/5.0.8

# 设置 OpenMP 线程数
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# 进入工作目录
cd $SLURM_SUBMIT_DIR

# 编译程序
echo "Compiling..."
make clean
make

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "=========================================="
echo "Running N-body simulation"
echo "Number of bodies: 500000"
echo "MPI processes: $SLURM_NTASKS"
echo "OpenMP threads per process: $OMP_NUM_THREADS"
echo "Total threads: $((SLURM_NTASKS * OMP_NUM_THREADS))"
echo "Start time: $(date)"
echo "=========================================="

# 运行程序并记录时间
start_time=$(date +%s.%N)
srun ./nbody_bh 500000
end_time=$(date +%s.%N)

elapsed=$(echo "$end_time - $start_time" | bc)

echo "=========================================="
echo "Job completed at $(date)"
echo "Wall clock time: $elapsed seconds"
echo "=========================================="
