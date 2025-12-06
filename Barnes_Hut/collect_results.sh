#!/bin/bash
# 收集强扩展性测试结果

echo "=========================================="
echo "Strong Scaling Results Summary"
echo "Problem size: 500000 particles"
echo "=========================================="
echo ""

printf "%-6s %-6s %-6s %-8s %-12s %-18s %-15s\n" "Nodes" "Tasks" "CPUs/T" "Total" "Time(s)" "Time/iter(s)" "Speedup"
echo "------------------------------------------------------------------------------------------------"

# 基准：1进程1线程（如果存在）
baseline_time=""
if [ -f "nbody_1n_1t_1c_173.out" ] && grep -q "Time per iteration:" "nbody_1n_1t_1c_173.out"; then
    baseline_time=$(grep "Time per iteration:" nbody_1n_1t_1c_173.out | awk '{print $4}')
fi

# 如果没有基准，使用1进程8线程作为基准
if [ -z "$baseline_time" ] && [ -f "nbody_1n_1t_8c_176.out" ]; then
    baseline_time=$(grep "Time per iteration:" nbody_1n_1t_8c_176.out | awk '{print $4}')
fi

for f in nbody_*n_*t_*c_*.out; do
    if [ ! -f "$f" ] || [ ! -s "$f" ]; then
        continue
    fi
    
    # 只处理有完整运行结果的（包含Time per iteration）
    if ! grep -q "Time per iteration:" "$f"; then
        continue
    fi
    
    job_id=$(grep "Job ID:" "$f" | head -1 | awk '{print $3}')
    nodes=$(grep "Nodes:" "$f" | head -1 | awk '{print $2}')
    tasks=$(grep "Total Tasks:" "$f" | head -1 | awk '{print $3}')
    cpus_per_task=$(grep "CPUs per Task:" "$f" | head -1 | awk '{print $4}')
    total_cpus=$(grep "Total CPUs:" "$f" | head -1 | awk '{print $3}')
    time_per_iter=$(grep "Time per iteration:" "$f" | awk '{print $4}')
    wall_time=$(grep "Wall clock time:" "$f" | awk '{print $4}')
    
    if [ -n "$time_per_iter" ] && [ -n "$baseline_time" ]; then
        speedup=$(echo "scale=2; $baseline_time / $time_per_iter" | bc)
        printf "%-6s %-6s %-6s %-8s %-12s %-18s %-15s\n" \
            "$nodes" "$tasks" "$cpus_per_task" "$total_cpus" "$wall_time" "$time_per_iter" "$speedup"
    else
        printf "%-6s %-6s %-6s %-8s %-12s %-18s %-15s\n" \
            "$nodes" "$tasks" "$cpus_per_task" "$total_cpus" "$wall_time" "$time_per_iter" "N/A"
    fi
done | sort -k4 -n

echo ""
echo "=========================================="
echo "Efficiency Analysis"
echo "=========================================="
echo ""

# 计算效率（相对于单核）
if [ -n "$baseline_time" ]; then
    echo "Baseline (1 process, 8 threads): $baseline_time seconds/iteration"
    echo ""
    printf "%-6s %-6s %-6s %-8s %-15s %-15s\n" "Nodes" "Tasks" "CPUs/T" "Total" "Speedup" "Efficiency(%)"
    echo "----------------------------------------------------------------------------"
    
    for f in nbody_*n_*t_*c_*.out; do
        if [ ! -f "$f" ] || ! grep -q "Time per iteration:" "$f"; then
            continue
        fi
        
        nodes=$(grep "Nodes:" "$f" | head -1 | awk '{print $2}')
        tasks=$(grep "Total Tasks:" "$f" | head -1 | awk '{print $3}')
        cpus_per_task=$(grep "CPUs per Task:" "$f" | head -1 | awk '{print $4}')
        total_cpus=$(grep "Total CPUs:" "$f" | head -1 | awk '{print $3}')
        time_per_iter=$(grep "Time per iteration:" "$f" | awk '{print $4}')
        
        if [ -n "$time_per_iter" ] && [ -n "$baseline_time" ]; then
            speedup=$(echo "scale=2; $baseline_time / $time_per_iter" | bc)
            efficiency=$(echo "scale=1; $speedup * 100 / $total_cpus" | bc)
            printf "%-6s %-6s %-6s %-8s %-15s %-15s\n" \
                "$nodes" "$tasks" "$cpus_per_task" "$total_cpus" "$speedup" "$efficiency"
        fi
    done | sort -k4 -n
fi

