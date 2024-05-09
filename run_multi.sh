#!/bin/bash

set -vx

CUDA="0,1,2,3,4"
NUM_GPUS=5

{
    CUDA_VISIBLE_DEVICES=${CUDA} bash train_multi.sh ${NUM_GPUS} \
        --task_name='SetPre4DEE_new' \
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
        --parallel_decorate \
        --re_label_map_path ./Data/label_map.json \
        --raat True \
        --raat_path_mem True \
        --num_relation 18
}
