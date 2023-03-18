#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

train_base_dir="/mnt/cephfs/hjh/train_record/images/text2image/controlnet"
dataset_base_dir="${train_base_dir}/dateset/fill50k"
train_out_dir="${train_base_dir}/train_out_fill50k"

mkdir -p ${train_out_dir}

sd_ini_file="${train_out_dir}/control_sd15_ini.ckpt"

#-------------------------------------------
# 初始化模型
#-------------------------------------------

if [ ! -f "${sd_ini_file}" ]; then

  sd_model_file="/mnt/cephfs/hjh/train_record/images/text2image/stable-diffusion/v1-5-pruned.ckpt"
  python tool_add_control.py ${sd_model_file} ${sd_ini_file}

  echo "----------------------"
  echo "save sd-ini file to :${sd_ini_file}"
  echo "----------------------"

else
  echo "-------- using sd-ini file:${sd_ini_file} -------"
fi

#-------------------------------------------
# 开始训练
#-------------------------------------------

resume_path=${sd_ini_file}
batch_size=4
logger_freq=300
cldm_config_file="$(pwd)/models/cldm_v15.yaml"
num_gpu=1
num_workers=0
prompt_file="${dataset_base_dir}/prompt.json"
data_dir=${dataset_base_dir}

CUDA_VISIBLE_DEVICES=3 \
  python -u tutorial_train.py \
  ${resume_path} \
  ${batch_size} \
  ${logger_freq} \
  ${cldm_config_file} \
  ${num_gpu} \
  ${num_workers} \
  ${prompt_file} \
  ${data_dir}