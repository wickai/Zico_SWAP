#!/bin/bash

# 创建日志文件夹
mkdir -p logs/gridsearch

# 定义超参数列表
generations_list=(300)
population_list=(20 40 60 80)

# size 
# 468 for 450m
# 625 for 600m
max_flops_cnn=625
export max_flops_cnn
echo "Max model size: $max_flops_cnn MFLOPs"

# Python 脚本路径
SCRIPT="src/search_imagenet.py"

# 结果 CSV 文件
RESULT_CSV="logs/gridsearch/results.csv"
echo "population_size,n_generations,swap_fitness" > "$RESULT_CSV"

# 构建所有任务组合
COMBINATIONS=()
for generations in "${generations_list[@]}"; do
    for population in "${population_list[@]}"; do
        COMBINATIONS+=("$population $generations")
    done
done

# 并行执行任务函数
run_task() {
    population=$1
    generations=$2
    gpu_id=$3

    LOGFILE="logs/gridsearch/log_pop${population}_gen${generations}.log"

    echo "Running pop=$population, gen=$generations on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT" \
        --population_size "$population" \
        --n_generations "$generations" \
        --device cuda \
        --num_classes 1000 \
        --log_path logs/gridsearch \
        --max_flops_cnn "$max_flops_cnn"\
        > "$LOGFILE" 2>&1

    fitness=$(grep "SWAP fitness=" "$LOGFILE" | tail -1 | sed 's/.*SWAP fitness=\([0-9.]*\).*/\1/')
    echo "$population,$generations,$fitness" >> "$RESULT_CSV"
}

export -f run_task
export SCRIPT RESULT_CSV

# 显式 GPU id 分配：假设你有4张卡
GPU_IDS=(0 1 2 3)
NUM_GPUS=${#GPU_IDS[@]}
NUM_TASKS=${#COMBINATIONS[@]}

# 为每个任务分配 GPU 并用 parallel 执行
for ((i=0; i<NUM_TASKS; i++)); do
    population=$(echo "${COMBINATIONS[i]}" | cut -d' ' -f1)
    generations=$(echo "${COMBINATIONS[i]}" | cut -d' ' -f2)
    gpu_id=${GPU_IDS[$((i % NUM_GPUS))]}
    echo "$population $generations $gpu_id"
done | parallel --colsep ' ' -j $NUM_GPUS run_task {1} {2} {3}

# 显示最终结果
echo -e "\n==== Grid Search Summary ===="
column -s, -t "$RESULT_CSV" | sort -k3 -nr
