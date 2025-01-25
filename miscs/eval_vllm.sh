TASK="mathvision_test"
MODEL_VERSION="ckpts/Qwen2-VL-2B-Instruct"
MODALITIES=image
export HF_ENDPOINT="https://hf-mirror.com"
export OPENAI_API_URL="http://0.0.0.0:8000/v1/chat/completions"
TASK_SUFFIX="${TASK//,/_}"

python -m lmms_eval \
    --model vllm \
    --model_args model_version=$MODEL_VERSION,modality=$MODALITIES,limit_img_num=1,tensor_parallel_size=4\
    --tasks $TASK \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ \
    --verbosity DEBUG
