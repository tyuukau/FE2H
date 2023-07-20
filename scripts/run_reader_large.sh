#!/bin/bash
echo "----------------------------------------------------"
# albert-xxlarge-v2
#echo "start to sleep"
#date
#sleep 1h
#echo "start to train model..."
export CUDA_VISIBLE_DEVICES="0,1,2"

BERT_MODEL=arvalinno/albert-base-v2-finetuned-squad
#albert-xxlarge-v2
MODEL_NAME=AlbertForQuestionAnsweringQANetWithMask

cd reader_src/origin_reader

PRETRAIN_LOG=20220125_run_qanet_pretrain_bs12_lr10_with_mask
PRETRAIN_TRAIN_FILE=../../data/hotpot_data/hotpot_labeled_data_squad.json
PRETRAIN_DEV_FILE=../../data/hotpot_data/hotpot_dev_labeled_data_v3.json
PRETRAIN_CACHE=../../data/cache/20220125_run_qanet_pretrain_bs12_lr10_with_mask
PRETRAIN_DIR=../../data/checkpoints/20220125_run_qanet_pretrain_bs12_lr10_with_mask
PRETRAIN_SUPPORTING_FILE=../../data/hotpot_data/train_golden.json

# truly train setting
TRAIN_DIR=../../data/checkpoints/20220125_run_qanet_pretrain_bs12_lr10_with_mask_step
TRAIN_TRAIN_FILE=../../data/hotpot_data/hotpot_train_labeled_data_v3.json
TRAIN_DEV_FILE=../../data/hotpot_data/hotpot_dev_labeled_data_v3.json
TRAIN_LOG=20220125_run_qanet_pretrain_bs12_lr10_with_mask_step
TRAIN_CACHE=../../data/cache/20220125_run_qanet_pretrain_bs12_lr10_with_mask_step
TRAIN_SUPPORTING_FILE=../../data/hotpot_data/train_golden.json

DEV_SUPPORTING_FILE=../../data/hotpot_data/dev_golden.json

# python -u origin_reader_model.py \
#   --bert_model $BERT_MODEL \
#   --output_dir  $PRETRAIN_DIR \
#   --model_name $MODEL_NAME \
#   --log_prefix $PRETRAIN_LOG \
#   --overwrite_result True \
#   --train_file $PRETRAIN_TRAIN_FILE \
#   --dev_file $PRETRAIN_DEV_FILE \
#   --train_supporting_para_file $PRETRAIN_SUPPORTING_FILE \
#   --dev_supporting_para_file ../../data/selector_data/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
#   --feature_cache_path $PRETRAIN_CACHE \
#   --train_batch_size 12 \
#   --gradient_accumulation_steps 1 \
#   --local_rank -1 \
#   --save_all_steps True \
#   --learning_rate 1e-5 \
#   --val_batch_size 12 \
#   --save_model_step 5000 \
#   --num_train_epochs 3.0
# echo "----------------------------------------------------"
# echo "pretrain done!"

python -u origin_reader_model.py \
  --bert_model $BERT_MODEL \
  --output_dir $TRAIN_DIR \
  --model_name $MODEL_NAME \
  --log_prefix  $TRAIN_LOG \
  --overwrite_result True \
  --train_file $TRAIN_TRAIN_FILE \
  --dev_file  $TRAIN_DEV_FILE \
  --train_supporting_para_file $TRAIN_SUPPORTING_FILE \
  --dev_supporting_para_file ../../data/selector_data/20211217_second_hop_electra_base_just_paragraph_selector_12_value_setting_result/dev_related.json \
  --feature_cache_path $TRAIN_CACHE \
  --train_batch_size 12 \
  --warmup_proportion 0.1 \
  --gradient_accumulation_steps 1 \
  --local_rank -1 \
  --save_all_steps True \
  --learning_rate 1e-5 \
  --val_batch_size 12 \
  --save_model_step 500 \
  --num_train_epochs 3.0
# --checkpoint_path $PRETRAIN_DIR \
echo "----------------------------------------------------"

echo "train done!"
