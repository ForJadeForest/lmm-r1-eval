
# pip3 install vllm
# pip3 install qwen_vl_utils

# cd ~/prod/lmms-eval-public
# pip3 install -e .
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=DEBUG
export OPENAI_API_URL="https://api.openai.com/v1/chat/completions"
export OPENAI_API_KEY="sk-proj-1234567890"
export BASE_URL="https://api.openai.com/v1"

python3 -m lmms_eval \
    --model qwen2_5_vl_vllm \
    --model_args model_version=Qwen/Qwen2.5-VL-7B-Instruct,tensor_parallel_size=1,gpu_memory_utilization=0.8,place_visual_first=True \
    --tasks mathverse_testmini,mathvista_testmini_cot,mathvision_test \
    --batch_size 64 \
    --log_samples \
    --log_samples_suffix vllm \
    --output_path ./logs \
    --limit=64 \
    --gen_kwargs "temperature=0,top_k=-1,top_p=1,max_new_tokens=8192"
