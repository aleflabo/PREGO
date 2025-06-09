#!/usr/bin/env bash

# PYTHONPATH=. torchrun --nproc_per_node 2 ./src/models/llama_meta.py --ckpt_dir=/media/ssd/usr/edo/llama/llama-2-13b --tokenizer_path=/media/ssd/usr/edo/llama/tokenizer.model --max_seq_len=2048 --max_batch_size=6 --temperature=0 --num_samples=5 --max_gen_len=4 --use_gt --dataset=assembly --type_prompt=num        
PYTHONPATH=. torchrun --nproc_per_node 1 ./src/models/llama_meta.py --ckpt_dir=/media/ssd/usr/edo/llama/llama-2-7b --tokenizer_path=/media/ssd/usr/edo/llama/tokenizer.model --max_seq_len=2048 --max_batch_size=6 --temperature=0.6 --num_samples=5 --max_gen_len=8 --dataset=assembly --type_prompt=emoji
# --use_gt 
PYTHONPATH=. torchrun --nproc_per_node 2 ./src/models/llama_meta.py --ckpt_dir=/media/ssd/usr/edo/llama/llama-2-14b --tokenizer_path=/media/ssd/usr/edo/llama/tokenizer.model --max_seq_len=2048 --max_batch_size=6 --temperature=0.6 --num_samples=5 --max_gen_len=8 --dataset=assembly --type_prompt=emoji