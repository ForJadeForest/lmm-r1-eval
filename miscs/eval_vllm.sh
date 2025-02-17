TASK="mathverse_testmini"
MODEL_VERSION="ckpts/Qwen2.5-VL-3B-Instruct-math_rloo"
MODALITIES=image
export HF_ENDPOINT="https://hf-mirror.com"
export OPENAI_API_URL="http://0.0.0.0:8000/v1/chat/completions"
export VLLM_USE_V1=1
TASK_SUFFIX="${TASK//,/_}"

python -m lmms_eval \
    --model vllm \
    --model_args model_version=$MODEL_VERSION,modality=$MODALITIES,limit_img_num=5,tensor_parallel_size=4,system_prompt_type=r1\
    --tasks $TASK \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ \
    --verbosity DEBUG
