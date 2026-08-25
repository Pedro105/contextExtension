| model | gpu | ctx_len | kv_cache_gib_per_seq | kv_limited_concurrency | scheduler_limited_concurrency | effective_concurrency | binding |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA L4 | 4096 | 0.7500 | 28 | 256 | 28 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA L4 | 8192 | 1.5000 | 14 | 256 | 14 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA L4 | 16384 | 3.0000 | 7 | 256 | 7 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA L4 | 32768 | 6.0000 | 3 | 256 | 3 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA L4 | 65536 | 12.0000 | 1 | 256 | 1 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA A100 80GB | 4096 | 0.7500 | 96 | 256 | 96 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA A100 80GB | 8192 | 1.5000 | 48 | 256 | 48 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA A100 80GB | 16384 | 3.0000 | 24 | 256 | 24 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA A100 80GB | 32768 | 6.0000 | 12 | 256 | 12 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA A100 80GB | 65536 | 12.0000 | 6 | 256 | 6 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA H100 80GB | 4096 | 0.7500 | 96 | 256 | 96 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA H100 80GB | 8192 | 1.5000 | 48 | 256 | 48 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA H100 80GB | 16384 | 3.0000 | 24 | 256 | 24 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA H100 80GB | 32768 | 6.0000 | 12 | 256 | 12 | kv |
| HuggingFaceTB/SmolLM2-1.7B | NVIDIA H100 80GB | 65536 | 12.0000 | 6 | 256 | 6 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA L4 | 4096 | 0.1562 | 138 | 256 | 138 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA L4 | 8192 | 0.3125 | 69 | 256 | 69 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA L4 | 16384 | 0.6250 | 34 | 256 | 34 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA L4 | 32768 | 1.2500 | 17 | 256 | 17 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA L4 | 65536 | 2.5000 | 8 | 256 | 8 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA A100 80GB | 4096 | 0.1562 | 460 | 256 | 256 | scheduler |
| HuggingFaceTB/SmolLM2-360M | NVIDIA A100 80GB | 8192 | 0.3125 | 230 | 256 | 230 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA A100 80GB | 16384 | 0.6250 | 115 | 256 | 115 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA A100 80GB | 32768 | 1.2500 | 57 | 256 | 57 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA A100 80GB | 65536 | 2.5000 | 28 | 256 | 28 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA H100 80GB | 4096 | 0.1562 | 460 | 256 | 256 | scheduler |
| HuggingFaceTB/SmolLM2-360M | NVIDIA H100 80GB | 8192 | 0.3125 | 230 | 256 | 230 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA H100 80GB | 16384 | 0.6250 | 115 | 256 | 115 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA H100 80GB | 32768 | 1.2500 | 57 | 256 | 57 | kv |
| HuggingFaceTB/SmolLM2-360M | NVIDIA H100 80GB | 65536 | 2.5000 | 28 | 256 | 28 | kv |
| HuggingFaceTB/SmolLM2-135M | NVIDIA L4 | 4096 | 0.0879 | 245 | 256 | 245 | kv |
| HuggingFaceTB/SmolLM2-135M | NVIDIA L4 | 8192 | 0.1758 | 122 | 256 | 122 | kv |
| HuggingFaceTB/SmolLM2-135M | NVIDIA L4 | 16384 | 0.3516 | 61 | 256 | 61 | kv |
| HuggingFaceTB/SmolLM2-135M | NVIDIA L4 | 32768 | 0.7031 | 30 | 256 | 30 | kv |
| HuggingFaceTB/SmolLM2-135M | NVIDIA L4 | 65536 | 1.4062 | 15 | 256 | 15 | kv |
| HuggingFaceTB/SmolLM2-135M | NVIDIA A100 80GB | 4096 | 0.0879 | 819 | 256 | 256 | scheduler |
| HuggingFaceTB/SmolLM2-135M | NVIDIA A100 80GB | 8192 | 0.1758 | 409 | 256 | 256 | scheduler |
| HuggingFaceTB/SmolLM2-135M | NVIDIA A100 80GB | 16384 | 0.3516 | 204 | 256 | 204 | kv |
| HuggingFaceTB/SmolLM2-135M | NVIDIA A100 80GB | 32768 | 0.7031 | 102 | 256 | 102 | kv |
| HuggingFaceTB/SmolLM2-135M | NVIDIA A100 80GB | 65536 | 1.4062 | 51 | 256 | 51 | kv |
| HuggingFaceTB/SmolLM2-135M | NVIDIA H100 80GB | 4096 | 0.0879 | 819 | 256 | 256 | scheduler |
| HuggingFaceTB/SmolLM2-135M | NVIDIA H100 80GB | 8192 | 0.1758 | 409 | 256 | 256 | scheduler |
| HuggingFaceTB/SmolLM2-135M | NVIDIA H100 80GB | 16384 | 0.3516 | 204 | 256 | 204 | kv |
| HuggingFaceTB/SmolLM2-135M | NVIDIA H100 80GB | 32768 | 0.7031 | 102 | 256 | 102 | kv |
| HuggingFaceTB/SmolLM2-135M | NVIDIA H100 80GB | 65536 | 1.4062 | 51 | 256 | 51 | kv |
| Qwen/Qwen2.5-1.5B | NVIDIA L4 | 4096 | 0.1094 | 197 | 256 | 197 | kv |
| Qwen/Qwen2.5-1.5B | NVIDIA L4 | 8192 | 0.2188 | 98 | 256 | 98 | kv |
| Qwen/Qwen2.5-1.5B | NVIDIA L4 | 16384 | 0.4375 | 49 | 256 | 49 | kv |
| Qwen/Qwen2.5-1.5B | NVIDIA L4 | 32768 | 0.8750 | 24 | 256 | 24 | kv |
| Qwen/Qwen2.5-1.5B | NVIDIA L4 | 65536 | 1.7500 | 12 | 256 | 12 | kv |
| Qwen/Qwen2.5-1.5B | NVIDIA A100 80GB | 4096 | 0.1094 | 658 | 256 | 256 | scheduler |
| Qwen/Qwen2.5-1.5B | NVIDIA A100 80GB | 8192 | 0.2188 | 329 | 256 | 256 | scheduler |
| Qwen/Qwen2.5-1.5B | NVIDIA A100 80GB | 16384 | 0.4375 | 164 | 256 | 164 | kv |
| Qwen/Qwen2.5-1.5B | NVIDIA A100 80GB | 32768 | 0.8750 | 82 | 256 | 82 | kv |
| Qwen/Qwen2.5-1.5B | NVIDIA A100 80GB | 65536 | 1.7500 | 41 | 256 | 41 | kv |
| Qwen/Qwen2.5-1.5B | NVIDIA H100 80GB | 4096 | 0.1094 | 658 | 256 | 256 | scheduler |
| Qwen/Qwen2.5-1.5B | NVIDIA H100 80GB | 8192 | 0.2188 | 329 | 256 | 256 | scheduler |
| Qwen/Qwen2.5-1.5B | NVIDIA H100 80GB | 16384 | 0.4375 | 164 | 256 | 164 | kv |
| Qwen/Qwen2.5-1.5B | NVIDIA H100 80GB | 32768 | 0.8750 | 82 | 256 | 82 | kv |
| Qwen/Qwen2.5-1.5B | NVIDIA H100 80GB | 65536 | 1.7500 | 41 | 256 | 41 | kv |
| google/gemma-3-1b-pt | NVIDIA L4 | 4096 | 0.0264 | 819 | 256 | 256 | scheduler |
| google/gemma-3-1b-pt | NVIDIA L4 | 8192 | 0.0420 | 514 | 256 | 256 | scheduler |
| google/gemma-3-1b-pt | NVIDIA L4 | 16384 | 0.0732 | 294 | 256 | 256 | scheduler |
| google/gemma-3-1b-pt | NVIDIA L4 | 32768 | 0.1357 | 159 | 256 | 159 | kv |
| google/gemma-3-1b-pt | NVIDIA L4 | 65536 | 0.2607 | 82 | 256 | 82 | kv |
| google/gemma-3-1b-pt | NVIDIA A100 80GB | 4096 | 0.0264 | 2730 | 256 | 256 | scheduler |
| google/gemma-3-1b-pt | NVIDIA A100 80GB | 8192 | 0.0420 | 1714 | 256 | 256 | scheduler |
| google/gemma-3-1b-pt | NVIDIA A100 80GB | 16384 | 0.0732 | 983 | 256 | 256 | scheduler |
| google/gemma-3-1b-pt | NVIDIA A100 80GB | 32768 | 0.1357 | 530 | 256 | 256 | scheduler |
| google/gemma-3-1b-pt | NVIDIA A100 80GB | 65536 | 0.2607 | 276 | 256 | 256 | scheduler |
| google/gemma-3-1b-pt | NVIDIA H100 80GB | 4096 | 0.0264 | 2730 | 256 | 256 | scheduler |
| google/gemma-3-1b-pt | NVIDIA H100 80GB | 8192 | 0.0420 | 1714 | 256 | 256 | scheduler |
| google/gemma-3-1b-pt | NVIDIA H100 80GB | 16384 | 0.0732 | 983 | 256 | 256 | scheduler |
| google/gemma-3-1b-pt | NVIDIA H100 80GB | 32768 | 0.1357 | 530 | 256 | 256 | scheduler |
| google/gemma-3-1b-pt | NVIDIA H100 80GB | 65536 | 0.2607 | 276 | 256 | 256 | scheduler |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA L4 | 4096 | 0.4062 | 53 | 256 | 53 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA L4 | 8192 | 0.8125 | 26 | 256 | 26 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA L4 | 16384 | 1.6250 | 13 | 256 | 13 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA L4 | 32768 | 3.2500 | 6 | 256 | 6 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA L4 | 65536 | 6.5000 | 3 | 256 | 3 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA A100 80GB | 4096 | 0.4062 | 177 | 256 | 177 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA A100 80GB | 8192 | 0.8125 | 88 | 256 | 88 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA A100 80GB | 16384 | 1.6250 | 44 | 256 | 44 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA A100 80GB | 32768 | 3.2500 | 22 | 256 | 22 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA A100 80GB | 65536 | 6.5000 | 11 | 256 | 11 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA H100 80GB | 4096 | 0.4062 | 177 | 256 | 177 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA H100 80GB | 8192 | 0.8125 | 88 | 256 | 88 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA H100 80GB | 16384 | 1.6250 | 44 | 256 | 44 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA H100 80GB | 32768 | 3.2500 | 22 | 256 | 22 | kv |
| mistralai/Ministral-3-3B-Base-2512 | NVIDIA H100 80GB | 65536 | 6.5000 | 11 | 256 | 11 | kv |
