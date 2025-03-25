#!/bin/bash

# Configuration parameters
export MODEL_CKPT="/path/to/your/model"  # Change to your model path

# System environment variables
export OPENAI_API_KEY="st-123"
export MODALITIES="image"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Default configuration parameters
LOG_DIR="./eval_results"
GPU_ID="0"  # GPU ID to use
DATASETS="mathverse_testmini,mathvision_test,mathvista_testmini_cot,mmstar"
# DATASETS="olympiadbench_test_en"
SYSTEM_PROMPT_TYPE="r1"  # System prompt type
PORT=8001  # API service port

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Print running parameters
echo -e "${BLUE}============================================${NC}"
echo -e "${BOLD}            Running Parameters              ${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "${CYAN}Model Path:        ${YELLOW}$MODEL_CKPT${NC}"
echo -e "${CYAN}GPU ID:            ${YELLOW}$GPU_ID${NC}"
echo -e "${CYAN}Datasets:          ${YELLOW}$DATASETS${NC}"
echo -e "${CYAN}Log Directory:     ${YELLOW}$LOG_DIR${NC}"
echo -e "${CYAN}System Prompt:     ${YELLOW}$SYSTEM_PROMPT_TYPE${NC}"
echo -e "${CYAN}Port:              ${YELLOW}$PORT${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Run evaluation task
run_evaluation() {
    local model_name=$(basename "$MODEL_CKPT")
    local output_path="${LOG_DIR}/${model_name}"
    
    echo -e "${GREEN}Starting evaluation for $DATASETS on model: ${BOLD}$MODEL_CKPT${NC}"
    echo -e "${GREEN}Results will be saved in: ${BOLD}$output_path${NC}"
    
    export OPENAI_API_URL="http://localhost:${PORT}/v1/chat/completions"
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    python -m lmms_eval \
        --model vllm \
        --model_args model_version=$MODEL_CKPT,modality=$MODALITIES,limit_img_num=5,tensor_parallel_size=1,system_prompt_type=$SYSTEM_PROMPT_TYPE,gpu_memory_utilization=0.7 \
        --tasks $DATASETS \
        --log_samples \
        --output_path $output_path \
        --verbosity DEBUG
        
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Evaluation completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ Evaluation failed${NC}"
        return 1
    fi
}

# Process dataset evaluation
process_datasets() {
    # Install correct dependencies based on dataset type
    if [[ "$DATASETS" == *"olympiadbench"* ]]; then
        echo -e "${BLUE}Installing dependencies for olympiad dataset${NC}"
        pip install antlr4-python3-runtime==4.11
    else
        echo -e "${BLUE}Installing dependencies for standard datasets${NC}"
        pip install antlr4-python3-runtime --upgrade
        pip install latex2sympy2 --upgrade
    fi
    
    # Run evaluation
    run_evaluation
}

# Install dependencies
echo -e "${BLUE}Installing necessary dependencies${NC}"
pip install qwen_vl_utils --upgrade
pip install loguru

# Start evaluation
process_datasets

echo -e "\n${GREEN}${BOLD}✓ Evaluation task completed${NC}\n"