#!/bin/bash
# Strong Scaling 测试脚本
# 固定问题规模：500000 个粒子
# 测试不同的进程数配置

N_BODIES=500000
BASE_DIR=$(pwd)

echo "=========================================="
echo "Strong Scaling Test"
echo "Problem size: $N_BODIES particles"
echo "=========================================="

# 测试配置：节点数 × 每节点进程数 × 每进程线程数
# 每个节点40核心，所以 进程数 × 线程数 = 40

configs=(
    "1 4 1"    # 1节点, 5进程, 8线程 = 40核心
    "1 4 2"   # 1节点, 10进程, 4线程 = 40核心
    "1 4 4"   # 1节点, 20进程, 2线程 = 40核心
    "1 4 8"   # 1节点, 40进程, 2线程 = 40核心
    "1 1 1"   # 1节点, 1进程, 1线程 = 1核心
    "1 1 2"   # 1节点, 1进程, 2线程 = 2核心
    "1 1 4"   # 1节点, 1进程, 4线程 = 4核心
    "1 1 8"   # 1节点, 1进程, 8线程 = 8核心
    "2 4 1"   # 2节点, 每节点5进程, 8线程 = 80核心
    "2 4 2"   # 2节点, 每节点10进程, 4线程 = 80核心
    "2 4 4"   # 2节点, 每节点20进程, 2线程 = 80核心
    "2 4 8"   # 2节点, 每节点40进程, 2线程 = 80核心
)

for config in "${configs[@]}"; do
    read nodes tasks_per_node cpus_per_task <<< "$config"
    total_tasks=$((nodes * tasks_per_node))
    
    echo ""
    echo "Submitting job: $nodes nodes, $tasks_per_node tasks/node, $cpus_per_task CPUs/task"
    echo "Total: $total_tasks MPI processes, $((total_tasks * cpus_per_task)) threads"
    
    # 创建临时SLURM脚本
    cat > nbody_bh_${nodes}n_${tasks_per_node}t_${cpus_per_task}c.slurm << EOF
#!/bin/bash
#SBATCH --job-name=nbody_${nodes}n_${tasks_per_node}t
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$tasks_per_node
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --time=02:00:00
#SBATCH --output=nbody_${nodes}n_${tasks_per_node}t_${cpus_per_task}c_%j.out
#SBATCH --error=nbody_${nodes}n_${tasks_per_node}t_${cpus_per_task}c_%j.err

echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Nodes: \$SLURM_JOB_NUM_NODES"
echo "Total Tasks: \$SLURM_NTASKS"
echo "Tasks per Node: \$SLURM_NTASKS_PER_NODE"
echo "CPUs per Task: \$SLURM_CPUS_PER_TASK"
echo "Total CPUs: \$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "=========================================="

# 加载必要的模块
module load StdEnv/2023
module load gcc/14.3
module load openmpi/5.0.8

# 设置 OpenMP 线程数
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# 进入工作目录
cd \$SLURM_SUBMIT_DIR

# 编译程序（使用profile版本支持gprof）
echo "Compiling with profiling support..."
make clean
make profile

if [ \$? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 确保文件系统同步（解决NFS延迟问题）
sync
sleep 1

# 验证可执行文件存在且可执行
if [ ! -f ./nbody_bh ]; then
    echo "ERROR: Executable nbody_bh not found after compilation!"
    ls -la
    exit 1
fi

if [ ! -x ./nbody_bh ]; then
    echo "WARNING: nbody_bh is not executable, fixing..."
    chmod +x ./nbody_bh
fi

echo "=========================================="
echo "Running N-body simulation"
echo "Number of bodies: $N_BODIES"
echo "MPI processes: \$SLURM_NTASKS"
echo "OpenMP threads per process: \$OMP_NUM_THREADS"
echo "Total threads: \$((SLURM_NTASKS * OMP_NUM_THREADS))"
echo "Start time: \$(date)"
echo "=========================================="

# 检查可执行文件是否存在
if [ ! -f ./nbody_bh ]; then
    echo "ERROR: Executable nbody_bh not found!"
    ls -la ./nbody_bh
    exit 1
fi

# 运行程序并记录时间
start_time=\$(date +%s.%N)
srun ./nbody_bh $N_BODIES
end_time=\$(date +%s.%N)

elapsed=\$(echo "\$end_time - \$start_time" | bc)

echo "=========================================="
echo "Job completed at \$(date)"
echo "Wall clock time: \$elapsed seconds"
echo "=========================================="

# 生成gprof报告（只分析rank 0的进程）
if [ \$SLURM_PROCID -eq 0 ] && [ -f gmon.out ]; then
    echo "Generating gprof profile..."
    gprof ./nbody_bh gmon.out > profile_rank0_\${SLURM_JOB_ID}.txt
    echo "Profile saved to profile_rank0_\${SLURM_JOB_ID}.txt"
fi
EOF
    
    # 提交作业
    sbatch nbody_bh_${nodes}n_${tasks_per_node}t_${cpus_per_task}c.slurm
    sleep 1
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Check status with: squeue -u \$USER"
echo "=========================================="

