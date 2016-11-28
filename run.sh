#!/usr/bin/env bash
source_path=$1
target_path=$2
inter_dir="inter"
split_dir="$inter_dir/split"
merge_dir="$inter_dir/merge"
root_dir="save"
load_path="$root_dir/34/basic-20000"
shared_path="$root_dir/34/shared.json"
out_path="$inter_dir/34.json"


load_path2="$root_dir/35/basic-20000"
shared_path2="$root_dir/35/shared.json"
out_path2="$inter_dir/35.json"

debug=False
python3 -m squad.prepro --mode single --single_path $source_path --debug $debug --target_dir $split_dir --glove_dir .
python3 -m squad.prepro --mode single --single_path $source_path --debug $debug --target_dir $merge_dir --glove_dir . --merge True

python3 -m basic.cli --data_dir $split_dir --nodump_eval --answer_path $out_path --load_path $load_path --shared_path $shared_path --draft $debug --eval_num_batches 0 --mode forward --batch_size 1 --len_opt --cluster --cpu_opt &
python3 -m basic.cli --data_dir $split_dir --nodump_eval --answer_path $out_path2 --load_path $load_path2 --shared_path $shared_path2 --draft $debug --eval_num_batches 0 --mode forward --batch_size 1 --len_opt --cluster --cpu_opt &
args="$out_path $out_path2"

for num in {0..7}
do
    load_path="$root_dir/7$num/basic-14000"
    shared_path="$root_dir/7$num/shared.json"
    out_path="$inter_dir/7$num.json"
    args="$args $out_path"
    python3 -m basic.cli --data_dir $merge_dir --nodump_eval --answer_path $out_path --load_path $load_path --shared_path $shared_path --draft $debug --eval_num_batches 0 --mode forward --batch_size 1 --len_opt --cluster --cpu_opt &

done
wait
python3 -m basic.combiner $target_path $args

