#!/bin/bash
#SBATCH --job-name=dist_strong
#SBATCH --output=dist_%j.out
#SBATCH --error=dist_%j.err
#SBATCH --time=02:00:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40        
#SBATCH --cpus-per-task=1           

# Strong scaling: 固定问题规模，改变进程数和线程数
N_BODIES=100000  # 固定粒子数量

# Strong scaling ranges
MPI_LIST="1 2 4 8 10 20 40 80"
OMP_LIST="1 2 4 8 10 20 40"

module load StdEnv/2023
module load gcc/14.3
module load openmpi/5.0.8

mkdir -p bin
rm -rf bin/*

echo "=========================================="
echo "Compiling nbody_bh_distributed.cpp..."
echo "=========================================="

mpic++ -std=c++14 -O3 -march=native -fopenmp -lm -o bin/nbody_bh_distributed nbody_bh_distributed.cpp

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation OK."
echo ""

echo "=========================================="
echo "Strong Scaling Test"
echo "Fixed problem size: $N_BODIES particles"
echo "=========================================="

for MPI_RANKS in $MPI_LIST; do
  for OMP_THREADS in $OMP_LIST; do

    TOTAL=$((MPI_RANKS * OMP_THREADS))

    # 必须不超过 80 total cores
    if [ $TOTAL -gt 80 ]; then
        continue
    fi

    echo ""
    echo "-------------------------------------------"
    echo "MPI ranks   : $MPI_RANKS"
    echo "OMP threads : $OMP_THREADS"
    echo "Total cores : $TOTAL"
    echo "Start time  : $(date)"

    export OMP_NUM_THREADS=$OMP_THREADS
    export OMP_PLACES=threads
    export OMP_PROC_BIND=spread

    # Run distributed N-body simulation
    start_time=$(date +%s.%N)
    mpirun --mca orte_leave_session_attached 1 -np $MPI_RANKS ./bin/nbody_bh_distributed $N_BODIES
    end_time=$(date +%s.%N)
    
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    echo "End time    : $(date)"
    echo "Elapsed time: $elapsed seconds"
    echo "-------------------------------------------"

  done
done

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="

