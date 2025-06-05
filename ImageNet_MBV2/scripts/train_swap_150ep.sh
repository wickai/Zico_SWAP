#!/bin/bash
cd "$(dirname "$0")"
set -e

# echo -e "Enter the metric"
# read metric
# echo -e "Enter the gpu"
# read gpu
# echo -e "Enter the seed"
# read seed
# echo "Run this script with metric=$metric, search gpu=$gpu, seed=$seed"

# if [ "$#" -lt 6 ] || [ "$#" -gt 6 ]; then
#     echo "$# is Illegal number of parameters."
#     echo "Usage: *.sh metric pop_size evo_iter seed num_workers init_method"
# 	exit 1
# fi

metric=swap
population_size=200
evolution_max_iter=5000
seed=123
num_workers=12
init=custom_kaiming
echo "Run this script with metric=$metric, population_size=$population_size, evolution_max_iter=$evolution_max_iter, seed=$seed, num_workers=$num_workers, init=$init"

cd ../

save_dir=./save_dir/${metric}_flops450M-searchbs64-pop${population_size}-iter${evolution_max_iter}-${seed}_28thApril_CIFAR10
mkdir -p ${save_dir}
evolution_max_iter=$(printf "%.0f" $evolution_max_iter)

resolution=32
epochs=150

python analyze_model.py \
  --input_image_size 32 \
  --num_classes 10 \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt

# CIFAR-10适用的训练参数
horovodrun -np 1 python train_image_classification.py --dataset cifar10 --num_classes 10 \
  --dist_mode single --workers_per_gpu ${num_workers} \
  --input_image_size ${resolution} --epochs ${epochs} --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 4e-5 --nesterov --weight_init ${init} \
  --label_smoothing \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  --use_se \
  --target_downsample_ratio 8 \
  --batch_size_per_gpu 128 --save_dir ${save_dir}/plain_training_epochs${epochs}_init-${init} \
  --datapath ./data


