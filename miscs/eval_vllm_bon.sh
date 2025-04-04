TASK="mathvision_test"
MODEL_VERSION="ckpts/Qwen2-VL-2B-Instruct"
RM_VERSION="ckpts/Qwen2-VL-2B-Instruct"
RM_HEAD="rm_head.pth" #nn.Linear(1536, 1)
MODALITIES=image
export HF_ENDPOINT="https://hf-mirror.com"
TASK_SUFFIX="${TASK//,/_}"

accelerate launch --num_processes 1 --main_process_port 30000 -m lmms_eval \
    --model vllm_bon \
    --model_args model_version=$MODEL_VERSION,rm_version=$RM_VERSION,rm_head=$RM_HEAD,n=8,mp=4,modality=$MODALITIES,limit_img_num=1,max_model_len=8000\
    --tasks $TASK \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ \
    --verbosity DEBUG
