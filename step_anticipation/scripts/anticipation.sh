#!/usr/bin/env bash
PYTHONPATH=. torchrun --nproc_per_node 1 ./src/models/chimera/chimera.py --ckpt_dir=/media/ssd/usr/edo/llama/llama-2-7b --tokenizer_path=/media/ssd/usr/edo/llama/tokenizer.model --max_seq_len=2048 --max_batch_size=6 --temperature=0 --num_samples=5 --max_gen_len=4 --skip_n=0 --use_gt --dataset=assembly --type_prompt=num        
