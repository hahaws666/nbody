#!/bin/bash
#SBATCH --job-name=nbody_bh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=nbody_bh_%j.out
#SBATCH --error=nbody_bh_%j.err

# 计算配置
# 2 nodes × 10 tasks/node × 4 cpus/task = 80 cores total
# 每个节点: 10 MPI 进程 × 4 OpenMP 线程 = 40 cores

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Total Tasks: $SLURM_NTASKS"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Total CPUs: $SLURM_CPUS_ON_NODE"
echo "=========================================="

# 加载必要的模块
module load StdEnv/2023
module load gcc/14.3
module load openmpi/5.0.8

# 设置 OpenMP 线程数（每个 MPI 进程）
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# 进入工作目录
cd $SLURM_SUBMIT_DIR

# 编译程序（使用profile版本支持gprof）
echo "Compiling with profiling support..."
make clean
make profile

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 运行参数（可以通过命令行参数修改）
N_BODIES=${1:-10000}  # 默认 10000 个粒子

echo "=========================================="
echo "Running N-body simulation"
echo "Number of bodies: $N_BODIES"
echo "MPI processes: $SLURM_NTASKS"
echo "OpenMP threads per process: $OMP_NUM_THREADS"
echo "Total threads: $((SLURM_NTASKS * OMP_NUM_THREADS))"
echo "=========================================="

# 运行程序
# 使用 srun 而不是 mpirun（在 SLURM 环境中推荐）
srun ./nbody_bh $N_BODIES

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="

# 生成gprof报告（只分析rank 0的进程）
if [ $SLURM_PROCID -eq 0 ] && [ -f gmon.out ]; then
    echo "Generating gprof profile..."
    gprof ./nbody_bh gmon.out > profile_rank0.txt
    echo "Profile saved to profile_rank0.txt"
fi

