TASK="mathvision_test"
MODEL_VERSION="ckpts/Qwen2-VL-2B-Instruct"
MODALITIES=image
export HF_ENDPOINT="https://hf-mirror.com"
TASK_SUFFIX="${TASK//,/_}"

accelerate launch --num_processes 1 --main_process_port 30000 -m lmms_eval \
    --model vllm \
    --model_args model_version=$MODEL_VERSION,modality=$MODALITIES,limit_img_num=1,max_model_len=4096\
    --tasks $TASK \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ \
    --verbosity DEBUG
