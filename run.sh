#!/usr/bin/env bash
source_path=$1
target_path=$2
model="basic"
run_id="01"
step="015000"
debug="False"
python3 -m squad.prepro_simple --mode single --single_path $source_path --debug $debug
answer_path="out/$model/$run_id/answer/single-$step.json"
python3 -m basic.cli --run_id $run_id --model $model --load_step $step --draft $debug --eval_num_batches 0 --mode forward --attention --batch_size 8 --use_glove_for_unk --known_if_glove --nofinetune --notraditional
cp $answer_path $target_path