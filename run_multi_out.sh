#!/bin/bash

set -vx

CUDA="0,1,2,3,4"
NUM_GPUS=5
task_name="SetPre4DEE_new" # 定义任务名称
outputDir="output/logs"
currTime=$(date +"%Y-%m-%d_%T")  # 获取当前时间
outputName="${outputDir}/run_${task_name}_${currTime}.log"

# 确保输出目录存在
if [ ! -d ${outputDir} ]; then
    mkdir -p ${outputDir}
fi

{
    CUDA_VISIBLE_DEVICES=${CUDA} nohup bash train_multi.sh ${NUM_GPUS} \
        --task_name=${task_name} \
        --use_bert=False \
        --skip_train=False \
        --train_nopair_sets=True  \
        --start_epoch=0 \
        --num_train_epochs=100 \
        --train_batch_size=32 \
        --gradient_accumulation_steps=4 \
        --learning_rate=1e-4 \
        --decoder_lr=2e-5 \
        --train_file_name='train.json' \
        --dev_file_name='dev.json' \
        --test_file_name='test.json' \
        --train_on_multi_events=True \
        --train_on_single_event=True \
        --num_event2role_decoder_layer=2 \
        --parallel_decorate > ${outputName} 2>&1 &
}
echo "Training started with PID $!"
