export CUDA_VISIBLE_DEVICES=2,3
conda activate eval
vllm serve "Qwen/Qwen2.5-14B-Instruct" \
    --dtype auto \
    --max-model-len 32000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --served-model-name gpt-3.5-turbo \
    --port 8001 \
    --api-key st-123
