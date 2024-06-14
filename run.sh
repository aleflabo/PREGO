#!/usr/bin/env bash
# torchrun --nproc_per_node 2 run_llama.py --ckpt_dir /media/ssd/usr/edo/llama/llama-2-13b/ --tokenizer_path /media/ssd/usr/edo/llama/tokenizer.model --max_seq_len 2048 --max_batch_size 6 --mode correct --temperature 0.6 --exp LS --num_samples 5 --max_gen_len 4

# PYTHONPATH=. torchrun --nproc_per_node 1 src/models/chimera/chimera.py --ckpt_dir /leonardo_work/IscrC_DbStM/data/llama2/llama-2-7b --tokenizer_path /leonardo_work/IscrC_DbStM/data/llama2/tokenizer.model --max_seq_len 2048 --max_batch_size 6 --temperature 0.6 --num_samples 5 --max_gen_len 4

# * Loki
# PYTHONPATH=. torchrun --nproc_per_node 1 src/models/chimera/chimera.py --ckpt_dir /media/ssd/usr/edo/llama/llama-2-7b  --tokenizer_path /media/ssd/usr/edo/llama/tokenizer.model --max_seq_len 3000 --max_batch_size 6 --temperature 0.6 --num_samples 5 --max_gen_len 4 --type_prompt=num
PYTHONPATH=. torchrun --nproc_per_node 1 src/models/chimera/chimera.py --ckpt_dir /media/ssd/usr/edo/llama/llama-2-7b  --tokenizer_path /media/ssd/usr/edo/llama/tokenizer.model --max_seq_len 3000 --max_batch_size 6 --temperature 0.6 --num_samples 5 --max_gen_len 8 --type_prompt=emoji # --use_gt --dataset=epictents
# PYTHONPATH=. torchrun --nproc_per_node 1 src/models/chimera/chimera.py --ckpt_dir /media/ssd/usr/edo/llama/llama-2-7b  --tokenizer_path /media/ssd/usr/edo/llama/tokenizer.model --max_seq_len 3000 --max_batch_size 6 --temperature 0.6 --num_samples 5 --max_gen_len 8 --type_prompt=alpha 

# * CINECA
# PYTHONPATH=. torchrun --nproc_per_node 1 src/models/chimera/chimera.py --ckpt_dir /leonardo_work/IscrC_DbStM/data/llama2/llama-2-7b --tokenizer_path /leonardo_work/IscrC_DbStM/data/llama2/tokenizer.model --max_seq_len 4096 --max_batch_size 6 --temperature 0.6 --num_samples 1 --max_gen_len 8 --type_prompt=emoji --dataset=assembly

# nvidia-smi
# python -c "import torch; print('---'); print(torch.cuda.device_count())" 

