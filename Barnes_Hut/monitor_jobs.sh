#!/bin/bash
# 监控作业状态

echo "Current job status:"
squeue -u $USER

echo ""
echo "Waiting for jobs to complete..."
echo "Press Ctrl+C to stop monitoring"

while true; do
    running=$(squeue -u $USER -h | wc -l)
    if [ $running -eq 0 ]; then
        echo ""
        echo "All jobs completed!"
        break
    fi
    echo -n "."
    sleep 10
done

echo ""
echo "Analyzing results..."
./analyze_scaling.sh

