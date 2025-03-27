#!/bin/bash

# Configuration parameters - modify this section
export MODEL_CKPT="VLM-Reasoner/LMM-R1-MGT-PerceReason"
export GPU_ID="0"  # GPU ID to use

# Evaluation tasks

TASKS="custom|math_500|0|0,custom|gpqa:diamond|0|0"
LOG_DIR="./eval_results"

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'


# Install dependencies
echo -e "${BLUE}Installing necessary dependencies...${NC}"


pip install antlr4-python3-runtime==4.11
pip install git+https://github.com/huggingface/lighteval.git@ed084813e0bd12d82a06d9f913291fdbee774905
pip install vllm==0.7.3
pip install transformers==4.49.0

# Print running parameters
echo -e "${BLUE}============================================${NC}"
echo -e "${BOLD}            Running Parameters              ${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "${CYAN}Model Path:        ${YELLOW}$MODEL_CKPT${NC}"
echo -e "${CYAN}GPU ID:            ${YELLOW}$GPU_ID${NC}"
echo -e "${CYAN}Tasks:             ${YELLOW}$TASKS${NC}"
echo -e "${CYAN}Log Directory:     ${YELLOW}$LOG_DIR${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Run evaluation task
run_evaluation() {
    local model_dir=$MODEL_CKPT
    local model_name=$(basename "$model_dir")
    local output_path="${LOG_DIR}/${model_name}"

    
    echo -e "${GREEN}Starting evaluation for $TASKS on model: ${BOLD}$model_dir${NC}"
    echo -e "${GREEN}Results will be saved in: ${BOLD}$output_path${NC}"
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Set model parameters
    MODEL_ARGS="pretrained=$model_dir,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.75,data_parallel_size=1,generation_parameters={max_new_tokens:8192,temperature:0,top_p:1.0,repetition_penalty:1.05}"
    
    # Run evaluation
    CUDA_VISIBLE_DEVICES=$GPU_ID lighteval vllm $MODEL_ARGS "$TASKS" \
        --custom-tasks evaluate_text.py \
        --use-chat-template \
        --output-dir $output_path
        
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Evaluation completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ Evaluation failed${NC}"
        return 1
    fi
}


# Start evaluation
echo -e "${BLUE}Starting model evaluation...${NC}"
run_evaluation

# Print completion message
echo -e "\n${GREEN}${BOLD}✓ Evaluation task completed${NC}\n" 
