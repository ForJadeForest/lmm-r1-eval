# cd /path/to/lmms-eval


#!/bin/bash
export API_KEY="EMPTY"
export API_URL="http://29.232.243.41:8002/v1"

export OPENAI_API_URL="http://29.191.208.124:8000/v1/chat/completions"
export BASE_URL="http://29.232.243.41:8000/v1"
export OPENAI_API_KEY="EMPTY"
export MODEL_VERSION="Qwen2.5-72B-Instruct"


export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

TASK=mathvista_testmini_cot
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

which python
python -m lmms_eval \
    --model code_qwen2_5_vl \
    --model_args model_version=O3_English_33k,sandbox_url="http://29.171.150.10:8080"\
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ \
    --verbosity=DEBUG
