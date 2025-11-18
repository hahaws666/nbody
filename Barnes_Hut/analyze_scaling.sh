#!/bin/bash
# 分析 Strong Scaling 结果

echo "=========================================="
echo "Strong Scaling Analysis"
echo "=========================================="
echo ""

# 查找所有输出文件
output_files=(nbody_*_*.out)

if [ ${#output_files[@]} -eq 0 ]; then
    echo "No output files found!"
    exit 1
fi

echo "Configuration | Nodes | Tasks/Node | CPUs/Task | Total CPUs | Time (s) | Speedup | Efficiency"
echo "--------------------------------------------------------------------------------------------------------"

# 存储结果
declare -A results

for file in "${output_files[@]}"; do
    if [ ! -f "$file" ]; then
        continue
    fi
    
    # 从文件名提取配置
    if [[ $file =~ nbody_([0-9]+)n_([0-9]+)t_([0-9]+)c_ ]]; then
        nodes=${BASH_REMATCH[1]}
        tasks=${BASH_REMATCH[2]}
        cpus=${BASH_REMATCH[3]}
        total_cpus=$((nodes * tasks * cpus))
        
        # 提取时间
        time=$(grep "Wall clock time:" "$file" | awk '{print $4}')
        
        if [ -n "$time" ]; then
            results["$total_cpus"]="$nodes|$tasks|$cpus|$total_cpus|$time"
        fi
    fi
done

# 按总CPU数排序
sorted_cpus=($(printf '%s\n' "${!results[@]}" | sort -n))

# 找到基准（最小CPU数）
if [ ${#sorted_cpus[@]} -gt 0 ]; then
    baseline_cpus=${sorted_cpus[0]}
    baseline_time=$(echo "${results[$baseline_cpus]}" | cut -d'|' -f5)
    
    for cpus in "${sorted_cpus[@]}"; do
        IFS='|' read -r nodes tasks cpus_per_task total_cpus time <<< "${results[$cpus]}"
        
        if [ -n "$time" ] && [ -n "$baseline_time" ]; then
            speedup=$(echo "scale=2; $baseline_time / $time" | bc)
            efficiency=$(echo "scale=2; $speedup * $baseline_cpus / $total_cpus * 100" | bc)
            printf "%-12s | %5s | %10s | %9s | %9s | %7s | %7s | %9s%%\n" \
                "${nodes}n${tasks}t${cpus_per_task}c" "$nodes" "$tasks" "$cpus_per_task" "$total_cpus" "$time" "$speedup" "$efficiency"
        fi
    done
fi

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="

