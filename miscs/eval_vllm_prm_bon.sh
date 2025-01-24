TASK="mathvision_test"
MODEL_VERSION="ckpts/Qwen2-VL-2B-Instruct"
RM_VERSION="ckpts/Qwen2-VL-2B-Instruct"
RM_HEAD="prm_head.take_min" #def take_min(rm_version): return lambda x:x[...,0].min(dim=-1,keepdim=True)
MODALITIES=image
export HF_ENDPOINT="https://hf-mirror.com"
TASK_SUFFIX="${TASK//,/_}"

accelerate launch --num_processes 1 --main_process_port 30000 -m lmms_eval \
    --model vllm_bon \
    --model_args model_version=$MODEL_VERSION,rm_version=$RM_VERSION,rm_head=$RM_HEAD,n=8,mp=4,modality=$MODALITIES,limit_img_num=1,max_model_len=8000,step_tag_id=151652,returned_token_ids="10 12"\
    --tasks $TASK \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ \
    --verbosity DEBUG
