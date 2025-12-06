#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strong Scaling Analysis for Barnes-Hut N-body Simulation
"""

import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# 数据提取
results = []

# 读取所有输出文件
import glob
import os

for f in sorted(glob.glob("nbody_*n_*t_*c_*.out")):
    if not os.path.isfile(f) or os.path.getsize(f) == 0:
        continue
    
    with open(f, 'r') as file:
        content = file.read()
        
        # 只处理有完整运行结果的
        if "Time per iteration:" not in content:
            continue
        
        # 提取信息
        job_id_match = re.search(r'Job ID:\s+(\d+)', content)
        nodes_match = re.search(r'Nodes:\s+(\d+)', content)
        tasks_match = re.search(r'Total Tasks:\s+(\d+)', content)
        cpus_per_task_match = re.search(r'CPUs per Task:\s+(\d+)', content)
        total_cpus_match = re.search(r'Total CPUs:\s+(\d+)', content)
        time_per_iter_match = re.search(r'Time per iteration:\s+([\d.]+)', content)
        wall_time_match = re.search(r'Wall clock time:\s+([\d.]+)', content)
        
        if all([job_id_match, nodes_match, tasks_match, cpus_per_task_match, 
                total_cpus_match, time_per_iter_match, wall_time_match]):
            results.append({
                'job_id': int(job_id_match.group(1)),
                'nodes': int(nodes_match.group(1)),
                'tasks': int(tasks_match.group(1)),
                'cpus_per_task': int(cpus_per_task_match.group(1)),
                'total_cpus': int(total_cpus_match.group(1)),
                'time_per_iter': float(time_per_iter_match.group(1)),
                'wall_time': float(wall_time_match.group(1)),
                'file': f
            })

# 按总CPU数排序
results.sort(key=lambda x: x['total_cpus'])

# 找到基准（1进程8线程）
baseline = None
for r in results:
    if r['nodes'] == 1 and r['tasks'] == 1 and r['cpus_per_task'] == 8:
        baseline = r
        break

if not baseline:
    print("Warning: No baseline found (1 node, 1 task, 8 CPUs/task)")
    baseline = results[0] if results else None

if baseline:
    baseline_time = baseline['time_per_iter']
    print("=" * 80)
    print("Strong Scaling Analysis Results")
    print("=" * 80)
    print(f"Problem size: 500000 particles")
    print(f"Baseline: {baseline['nodes']} node(s), {baseline['tasks']} task(s), "
          f"{baseline['cpus_per_task']} CPU(s)/task = {baseline['total_cpus']} total CPUs")
    print(f"Baseline time: {baseline_time:.4f} seconds/iteration")
    print("=" * 80)
    print()
    
    # 计算加速比和效率
    print(f"{'Nodes':<6} {'Tasks':<6} {'CPUs/T':<7} {'Total':<6} {'Time/iter(s)':<15} "
          f"{'Speedup':<10} {'Efficiency(%)':<15}")
    print("-" * 80)
    
    for r in results:
        speedup = baseline_time / r['time_per_iter']
        efficiency = (speedup / r['total_cpus']) * 100
        print(f"{r['nodes']:<6} {r['tasks']:<6} {r['cpus_per_task']:<7} {r['total_cpus']:<6} "
              f"{r['time_per_iter']:<15.4f} {speedup:<10.2f} {efficiency:<15.1f}")
    
    print()
    print("=" * 80)
    print("Key Findings:")
    print("=" * 80)
    
    # 找到最佳配置
    best = min(results, key=lambda x: x['time_per_iter'])
    best_speedup = baseline_time / best['time_per_iter']
    best_efficiency = (best_speedup / best['total_cpus']) * 100
    
    print(f"Best performance: {best['nodes']} node(s), {best['tasks']} task(s), "
          f"{best['cpus_per_task']} CPU(s)/task")
    print(f"  Time per iteration: {best['time_per_iter']:.4f} seconds")
    print(f"  Speedup: {best_speedup:.2f}x")
    print(f"  Efficiency: {best_efficiency:.1f}%")
    print()
    
    # 找到最高效率配置
    best_eff = max(results, key=lambda x: (baseline_time / x['time_per_iter']) / x['total_cpus'])
    best_eff_speedup = baseline_time / best_eff['time_per_iter']
    best_eff_efficiency = (best_eff_speedup / best_eff['total_cpus']) * 100
    
    print(f"Best efficiency: {best_eff['nodes']} node(s), {best_eff['tasks']} task(s), "
          f"{best_eff['cpus_per_task']} CPU(s)/task")
    print(f"  Time per iteration: {best_eff['time_per_iter']:.4f} seconds")
    print(f"  Speedup: {best_eff_speedup:.2f}x")
    print(f"  Efficiency: {best_eff_efficiency:.1f}%")
    print()
    
    # 生成图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 加速比 vs 总CPU数
    ax1 = axes[0, 0]
    total_cpus = [r['total_cpus'] for r in results]
    speedups = [baseline_time / r['time_per_iter'] for r in results]
    ideal_speedups = total_cpus  # 理想线性加速
    
    ax1.plot(total_cpus, speedups, 'o-', label='Actual Speedup', linewidth=2, markersize=8)
    ax1.plot(total_cpus, ideal_speedups, '--', label='Ideal Linear Speedup', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Total CPUs', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Strong Scaling: Speedup vs Total CPUs', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 效率 vs 总CPU数
    ax2 = axes[0, 1]
    efficiencies = [(baseline_time / r['time_per_iter']) / r['total_cpus'] * 100 for r in results]
    
    ax2.plot(total_cpus, efficiencies, 'o-', color='green', linewidth=2, markersize=8)
    ax2.axhline(y=100, color='r', linestyle='--', label='Ideal Efficiency (100%)', alpha=0.7)
    ax2.set_xlabel('Total CPUs', fontsize=12)
    ax2.set_ylabel('Efficiency (%)', fontsize=12)
    ax2.set_title('Strong Scaling: Efficiency vs Total CPUs', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 时间/迭代 vs 总CPU数
    ax3 = axes[1, 0]
    times = [r['time_per_iter'] for r in results]
    
    ax3.plot(total_cpus, times, 'o-', color='red', linewidth=2, markersize=8)
    ax3.set_xlabel('Total CPUs', fontsize=12)
    ax3.set_ylabel('Time per Iteration (seconds)', fontsize=12)
    ax3.set_title('Strong Scaling: Time per Iteration vs Total CPUs', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. 加速比 vs 效率散点图
    ax4 = axes[1, 1]
    
    ax4.scatter(total_cpus, speedups, s=100, c=efficiencies, cmap='viridis', 
               edgecolors='black', linewidth=1.5, alpha=0.7)
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Efficiency (%)', fontsize=10)
    ax4.set_xlabel('Total CPUs', fontsize=12)
    ax4.set_ylabel('Speedup', fontsize=12)
    ax4.set_title('Speedup vs Total CPUs (colored by Efficiency)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('strong_scaling_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: strong_scaling_analysis.png")
    print("=" * 80)

else:
    print("No valid results found!")

