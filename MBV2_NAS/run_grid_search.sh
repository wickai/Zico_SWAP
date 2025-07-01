#!/bin/bash

# 创建日志文件夹
mkdir -p logs/gridsearch

# 定义超参数列表
generations_list=(200 300)
population_list=(20 40 60 80)

# 可执行的 Python 脚本路径
SCRIPT="src/search_imagenet.py"  # ← 替换为实际脚本名

# CSV 文件保存每个组合的结果
RESULT_CSV="logs/gridsearch/results.csv"
echo "population_size,n_generations,swap_fitness" > $RESULT_CSV

# 遍历组合
for generations in "${generations_list[@]}"; do
    for population in "${population_list[@]}"; do
        echo "Running with population_size=${population}, n_generations=${generations}"

        LOGFILE="logs/gridsearch/log_pop${population}_gen${generations}.log"

        # 运行 Python 脚本并记录日志
        python $SCRIPT \
            --population_size $population \
            --n_generations $generations \
            --device cuda \
            --num_classes 1000 \
            --log_path logs/gridsearch \
            > $LOGFILE 2>&1

        # 从日志中提取 SWAP fitness
        fitness=$(grep "SWAP fitness=" "$LOGFILE" | tail -1 | sed 's/.*SWAP fitness=\([0-9.]*\).*/\1/')

        # 记录结果到 CSV
        echo "$population,$generations,$fitness" >> $RESULT_CSV
    done
done

# 显示最终结果
echo -e "\n==== Grid Search Summary ===="
column -s, -t $RESULT_CSV | sort -k3 -nr
