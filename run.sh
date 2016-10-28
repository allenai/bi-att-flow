#!/usr/bin/env bash
source_path=$1
target_path=$2
inter_dir="inter"
root_dir="save"
load_path="$root_dir/34/basic-20000"
shared_path="$root_dir/34/shared.json"
out_path="$inter_dir/34.json"
load_path2="$root_dir/35/basic-20000"
shared_path2="$root_dir/35/shared.json"
out_path2="$inter_dir/35.json"
debug=False
python3 -m squad.prepro --mode single --single_path $source_path --debug $debug --target_dir $inter_dir --glove_dir .
python3 -m basic.cli --data_dir $inter_dir --nodump_eval --answer_path $out_path --load_path $load_path --shared_path $shared_path --draft $debug --eval_num_batches 0 --mode forward --batch_size 1 --bi --len_opt --cluster
python3 -m basic.cli --data_dir $inter_dir --nodump_eval --answer_path $out_path2 --load_path $load_path2 --shared_path $shared_path2 --draft $debug --eval_num_batches 0 --mode forward --batch_size 1 --bi --len_opt --cluster
python3 -m basic.combiner $target_path $out_path $out_path2

