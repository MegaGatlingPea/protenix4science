#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
code_directory = os.path.dirname(current_directory)
sys.path.append(code_directory)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUTLASS_PATH'] = '/home/megagatlingpea/workdir/protenix4science/cutlass'  # [Xujun] 添加cutlass路径，用于ds4sci
os.environ["LAYERNORM_TYPE"] = "fast_layernorm"
os.environ["USE_DEEPSPEED_EVO_ATTTENTION"] = "true"
project = "protenix_datacondition"
run_name = "data_condition"
base_dir = f"{code_directory}/output"

## related to training
dtype = "bf16"
use_wandb = False
diffusion_batch_size = 48 # 32
eval_interval = 400
log_interval = 1
checkpoint_interval = 400  # default = -1
ema_decay = 0.999  # default = -1.0
train_crop_size = 384 # 640  # default = 256
max_steps = 100000
warmup_steps = 2000  # default = 10
lr = 0.001  # default = 0.0018
data_train_sets = ["weightedPDB_before2109_wopb_nometalc_0925"]
data_test_sets = [
    "recentPDB_1536_sample384_0925",
    # "posebusters_0925",
]  # default = ["recentPDB_1536_sample384_0925"]
data_posebusters_0925_base_info_max_n_token = 768
num_dl_workers = 16  # if not debug else 0  # 0 for debug and 16 for training/inference

# finetune相关，从头训练忽略这块即可
# 下载以下文件
# "model_v0.2.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.2.0.pt" 到 ./release_data/checkpoint/model_v0.2.0.pt
checkpoint_path = os.path.join(code_directory, "release_data/checkpoint/model_v0.2.0.pt") ## for finetune
ema_checkpoint_path = os.path.join(code_directory, "release_data/checkpoint/model_v0.2.0.pt") ## for finetune
load_checkpoint_path = checkpoint_path  # '' for training and ${checkpoint_path} for finetune
load_ema_checkpoint_path = ema_checkpoint_path  # '' for training and ${checkpoint_path} for finetune

## also related to inference
N_cycle = 4  # 10 for inference
N_sample = 5  # 5 for both training and inference
sample_diffusion_N_step = 20  # 200 for inference

# [Xujun] 增加蛋白质主链生成的开关逻辑参数
# data: 对蛋白、核酸和分子进行mask，包括侧链原子、token和原子类型
# constraint: 计算condition token原子间的最大距离作为constraint，在pairformer中进行约束
# diffusion: 保留condition atom的坐标，在diffusion模块中进行约束
# all:包括了上述三种condition类型
data_condition = 'all'  # ['all', 'data', 'constraint', 'diffusion']
# [Xujun] END