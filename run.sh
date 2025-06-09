#!/usr/bin/env bash

PYTHONPATH=. python ./step_anticipation/src/models/chimera/llm_hf.py --model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --max_seq_len=2048 --max_batch_size=6 --temperature=0.6 --max_gen_len=5 --dataset=epictent --type_prompt=num --num_samples=5

# --use_gt
# --model_name=unsloth/Llama-3.2-1B
# --model_name=meta-llama/Llama-3.2-1B
# --model_name=meta-llama/Llama-2-7B-hf
