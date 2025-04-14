

gpus='4,5,6,7'
comma_count=$(echo "$gpus" | grep -o ',' | wc -l)
nproc_per_node=$((comma_count+1))
torchrun --standalone --nnodes=1 --nproc_per_node=${nproc_per_node} run_kws_train.py --gpus=${gpus}

