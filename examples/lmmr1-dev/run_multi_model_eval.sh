#!/bin/bash
# Script to run evaluations for multiple models in parallel across specified GPUs using named arguments.

# --- Environment Setup ---
# Activate conda environment if needed
cd /path/to/your/lmms-eval  # Change this to your lmms-eval directory if needed

# Ensure necessary packages are installed
pip install qwen_vl_utils --upgrade
pip install loguru


echo "Setting up environment variables..."
export HF_DATASETS_OFFLINE=1

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=WARN
export OPENAI_API_URL="http://localhost:8000/v1/chat/completions"
export BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="st-123"

# --- Default Configuration ---
BASE_MODEL_PATH="/path/to/your/model_ckpts" # Adjust as needed
LOG_DIR="./eval_results/main" # Define a log directory
DEFAULT_DATASETS="mathverse_testmini,mathvista_testmini_cot,mathvision_test" # Default: specific math datasets handled later
DEFAULT_SYSTEM_PROMPT_PATH="./scripts/new_r1_system_prompt.txt"
DEFAULT_OVERWRITE="false"
DEFAULT_LAST_STEP_ONLY="false"
DEFAULT_BATCH_SIZE=1000
DEFAULT_GPU_IDS="0"
DEFAULT_MODEL_KEYS="3B_baseline"


# Define model name dictionary (Maps shorthand key to actual model directory name under BASE_MODEL_PATH)
declare -A MODEL_NAMES=(
    ["3B_baseline"]="Qwen2.5-VL-3B-Instruct"
    ["7B_baseline"]="Qwen2.5-VL-7B-Instruct"
    # Add more model keys and corresponding directory names here
    # Example: ["my_model_key"]="my_model_directory_name"
)

# --- Argument Parsing with getopt ---
# Define short and long options
SHORT_OPTS="g:m:d:s:o:l:b:h"
LONG_OPTS="gpu_ids:,model_keys:,datasets:,system_prompt_path:,overwrite:,last_step_only:,batch_size:,help"

# Parse options
PARSED_OPTIONS=$(getopt -o $SHORT_OPTS --long $LONG_OPTS -n "$0" -- "$@")

# Check if getopt encountered an error
if [ $? -ne 0 ]; then
    echo "Error parsing options." >&2
    exit 1
fi

# Set the parsed options as the script's arguments
eval set -- "$PARSED_OPTIONS"

# Initialize variables with defaults
GPU_IDS="$DEFAULT_GPU_IDS"
MODEL_KEYS="$DEFAULT_MODEL_KEYS"
CUSTOM_DATASETS="$DEFAULT_DATASETS"
SYSTEM_PROMPT_PATH="$DEFAULT_SYSTEM_PROMPT_PATH"
OVERWRITE="$DEFAULT_OVERWRITE"
LAST_STEP_ONLY="$DEFAULT_LAST_STEP_ONLY"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"

# Function to display help message
usage() {
    echo "Usage: $0 --gpu_ids <ids> --model_keys <keys> [options]"
    echo ""
    echo "Required arguments:"
    echo "  -g, --gpu_ids <ids>           Comma-separated GPU IDs (e.g., "0,1,2,3")"
    echo "  -m, --model_keys <keys>       Comma-separated model keys defined in the script (e.g., "3B_baseline,k12_fre")"
    echo ""
    echo "Optional arguments:"
    echo "  -d, --datasets <datasets>     Comma-separated datasets (default: specific math datasets handled later)"
    echo "  -s, --system_prompt_path <path> Path to the system prompt file (default: none)"
    echo "  -o, --overwrite <true|false>  Whether to overwrite existing results (default: $DEFAULT_OVERWRITE)"
    echo "  -l, --last_step_only <true|false> Evaluate only the last checkpoint/completed model (default: $DEFAULT_LAST_STEP_ONLY)"
    echo "  -b, --batch_size <num>        Evaluation batch size per GPU (default: $DEFAULT_BATCH_SIZE)"
    echo "  -h, --help                    Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --gpu_ids "0,1" --model_keys "3B_baseline,k12_fre" --datasets "mathvista_testmini_cot,mmbench_dev_en" --system_prompt_path "./scripts/new_r1_system_prompt.txt" --overwrite true --batch_size 16"
    exit 1
}


# Process parsed options
while true; do
    case "$1" in
        -g|--gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        -m|--model_keys)
            MODEL_KEYS="$2"
            shift 2
            ;;
        -d|--datasets)
            CUSTOM_DATASETS="$2"
            shift 2
            ;;
        -s|--system_prompt_path)
            SYSTEM_PROMPT_PATH="$2"
            shift 2
            ;;
        -o|--overwrite)
            OVERWRITE=$(echo "$2" | tr '[:upper:]' '[:lower:]') # Convert to lowercase
             if [[ "$OVERWRITE" != "true" && "$OVERWRITE" != "false" ]]; then
                 echo "Error: Invalid value for --overwrite. Use 'true' or 'false'." >&2
                 usage
             fi
            shift 2
            ;;
        -l|--last_step_only)
             LAST_STEP_ONLY=$(echo "$2" | tr '[:upper:]' '[:lower:]') # Convert to lowercase
             if [[ "$LAST_STEP_ONLY" != "true" && "$LAST_STEP_ONLY" != "false" ]]; then
                 echo "Error: Invalid value for --last_step_only. Use 'true' or 'false'." >&2
                 usage
             fi
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
             # Basic check if it's a positive integer
             if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
                 echo "Error: Invalid value for --batch_size. Must be a positive integer." >&2
                 usage
             fi
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error! Unexpected option: $1" >&2
            usage
            ;;
    esac
done

# --- Validate Required Arguments ---
if [ -z "$GPU_IDS" ]; then
    echo "Error: --gpu_ids is required." >&2
    usage
fi
if [ -z "$MODEL_KEYS" ]; then
    echo "Error: --model_keys is required." >&2
    usage
fi


# --- Color Codes for Output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# --- Initial Setup & Validation ---
# Convert GPU IDs to array
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Convert model keys to array
IFS=',' read -r -a MODEL_KEY_ARRAY <<< "$MODEL_KEYS"

# Check if system prompt file exists
SYSTEM_PROMPT_ARG=""
if [ -n "$SYSTEM_PROMPT_PATH" ]; then
    if [ -f "$SYSTEM_PROMPT_PATH" ]; then
        SYSTEM_PROMPT_ARG="system_prompt=$SYSTEM_PROMPT_PATH"
        echo -e "${GREEN}Using system prompt file: $SYSTEM_PROMPT_PATH${NC}"
    else
        echo -e "${RED}Error: System prompt file not found at $SYSTEM_PROMPT_PATH${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}No system prompt file provided.${NC}"
fi


# Print Running Parameters
echo -e "${BLUE}============================================${NC}"
echo -e "${BOLD}            Running Parameters              ${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "${CYAN}Base Model Path:    ${YELLOW}$BASE_MODEL_PATH${NC}"
echo -e "${CYAN}Log Directory:     ${YELLOW}$LOG_DIR${NC}"
echo -e "${CYAN}GPU IDs:           ${YELLOW}$GPU_IDS${NC}"
echo -e "${CYAN}Number of GPUs:    ${YELLOW}$NUM_GPUS${NC}"
echo -e "${CYAN}Model Keys:        ${YELLOW}$MODEL_KEYS${NC}"
echo -e "${CYAN}Custom Datasets:   ${YELLOW}${CUSTOM_DATASETS:-Default Math Datasets handled later}${NC}"
echo -e "${CYAN}System Prompt:     ${YELLOW}${SYSTEM_PROMPT_PATH:-None}${NC}"
echo -e "${CYAN}Overwrite:         ${YELLOW}$OVERWRITE${NC}"
echo -e "${CYAN}Last Step Only:    ${YELLOW}$LAST_STEP_ONLY${NC}"
echo -e "${CYAN}Batch Size:        ${YELLOW}$BATCH_SIZE${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# --- Helper Functions ---

# Find checkpoint directories sorted numerically descending
find_checkpoints() {
    local dir="$1"
    # Look for directories matching global_step*_hf pattern
    find "$dir" -maxdepth 1 -type d -name "global_step*_hf" | sort -rV
}

# Check if a model directory contains .safetensors (indicating completion)
check_model_completed() {
    local model_dir="$1"
    # Check for any file ending in .safetensors
    if ls "$model_dir"/*.safetensors >/dev/null 2>&1; then
        return 0 # Completed
    else
        return 1 # Not completed
    fi
}

# --- Evaluation Function ---
# Runs a single evaluation task on a specific GPU
run_evaluation() {
    local model_dir=$1      # Full path to the model checkpoint or directory
    local model_name=$2     # Logical name of the model (from MODEL_NAMES)
    local gpu_id=$3         # GPU ID to use
    local is_last_ckpt=${4:-false} # Flag indicating if it's the final version/last checkpoint
    local datasets_to_run=$5 # Datasets for this specific run

    local step_name
    if [ "$is_last_ckpt" = true ]; then
        # Use "final" for completed models or the single checkpoint if LAST_STEP_ONLY=true
         if check_model_completed "$model_dir"; then
             step_name="final_completed"
         else
             step_name=$(basename "$model_dir") # Use the actual checkpoint name if root isn't complete
         fi
    else
        step_name=$(basename "$model_dir") # Use checkpoint directory name
    fi

    local output_path="${LOG_DIR}/${model_name}/${step_name}"

    # Check if results exist and handle overwrite logic
    if [ -d "$output_path" ] && [ "$OVERWRITE" != "true" ]; then
        echo -e "${YELLOW}[GPU $gpu_id] Skipping: Output directory already exists for $model_name/$step_name: $output_path${NC}"
        return 0
    elif [ -d "$output_path" ] && [ "$OVERWRITE" = "true" ]; then
        echo -e "${YELLOW}[GPU $gpu_id] Overwriting existing results in: $output_path${NC}"
        # Consider adding rm -rf "$output_path" here if you want a clean overwrite
    fi

    mkdir -p "$output_path"

    echo -e "${GREEN}[GPU $gpu_id] Starting evaluation for ${BOLD}$model_name ($step_name)${NC}"
    echo -e "${GREEN}[GPU $gpu_id] Model Path: ${BOLD}$model_dir${NC}"
    echo -e "${GREEN}[GPU $gpu_id] Datasets:   ${BOLD}$datasets_to_run${NC}"
    echo -e "${GREEN}[GPU $gpu_id] Output:     ${BOLD}$output_path${NC}"

    # Prepare model arguments string
    local model_args="model_version=$model_dir,tensor_parallel_size=1,gpu_memory_utilization=0.8,place_visual_first=True"
    if [ -n "$SYSTEM_PROMPT_ARG" ]; then
         model_args="$model_args,$SYSTEM_PROMPT_ARG"
    fi

    # Set CUDA device for this process
    export CUDA_VISIBLE_DEVICES=$gpu_id

    # Construct and run the evaluation command
    python3 -m lmms_eval \
        --model qwen2_5_vl_vllm \
        --model_args $model_args \
        --tasks "$datasets_to_run" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --log_samples_suffix "${model_name}_${step_name}" \
        --output_path "$output_path" \
        --gen_kwargs "temperature=0,top_k=-1,top_p=1,max_new_tokens=8192" # From new example

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[GPU $gpu_id] ✓ Evaluation completed successfully for $model_name ($step_name)${NC}"
    else
        echo -e "${RED}[GPU $gpu_id] ✗ Evaluation FAILED for $model_name ($step_name) (Exit Code: $exit_code)${NC}"
        # Optional: Create a failure marker file
        # touch "$output_path/EVALUATION_FAILED"
        return 1 # Propagate failure
    fi
    return 0
}

# --- Model Discovery ---
# Collect all model paths/checkpoints to be evaluated
declare -a ALL_MODELS_TO_EVALUATE
declare -a ALL_MODEL_NAMES_FOR_EVAL
declare -a IS_LAST_CKPT_FOR_EVAL

echo -e "${BLUE}Collecting models and checkpoints for evaluation...${NC}"

for model_key in "${MODEL_KEY_ARRAY[@]}"; do
    if [[ -z "${MODEL_NAMES[$model_key]}" ]]; then
        echo -e "${YELLOW}Warning: Model key '$model_key' not found in MODEL_NAMES. Skipping.${NC}"
        continue
    fi

    model_name="${MODEL_NAMES[$model_key]}"
    model_root_dir="$BASE_MODEL_PATH/$model_name"

    if [ ! -d "$model_root_dir" ]; then
        echo -e "${YELLOW}Warning: Model directory does not exist: $model_root_dir. Skipping '$model_name'.${NC}"
        continue
    fi

    echo -e "\n${CYAN}Processing model: ${BOLD}$model_name (Key: $model_key)${NC}"

    found_evaluable_version=false

    # 1. Check for completed model in the root directory
    if check_model_completed "$model_root_dir"; then
        echo -e "${GREEN}  Found completed model in root: $model_root_dir${NC}"
        ALL_MODELS_TO_EVALUATE+=("$model_root_dir")
        ALL_MODEL_NAMES_FOR_EVAL+=("$model_name")
        IS_LAST_CKPT_FOR_EVAL+=("true") # Mark as final version
        found_evaluable_version=true
        # If only evaluating the last step, we are done with this model key
        if [ "$LAST_STEP_ONLY" = "true" ]; then
            echo -e "${CYAN}  LAST_STEP_ONLY=true, skipping checkpoint search for $model_name.${NC}"
            continue # Move to the next model key
        fi
    else
         echo -e "${YELLOW}  No completed model found in root: $model_root_dir${NC}"
         # If LAST_STEP_ONLY=true, we still need to find the latest checkpoint below
    fi

    # 2. Check for checkpoints in the 'ckpt' subdirectory (only if not LAST_STEP_ONLY=true OR if root wasn't complete)
    ckpt_dir="$model_root_dir/ckpt"
    if [ -d "$ckpt_dir" ]; then
        echo -e "${CYAN}  Searching for checkpoints in: $ckpt_dir${NC}"
        CHECKPOINTS=($(find_checkpoints "$ckpt_dir"))

        if [ ${#CHECKPOINTS[@]} -gt 0 ]; then
            echo -e "${CYAN}  Found ${#CHECKPOINTS[@]} checkpoints.${NC}"
            if [ "$LAST_STEP_ONLY" = "true" ] && [ "$found_evaluable_version" = "false" ]; then
                # If only last step and root wasn't complete, evaluate only the latest checkpoint
                latest_ckpt=${CHECKPOINTS[0]} # Assumes find_checkpoints sorts descending
                echo -e "${GREEN}  LAST_STEP_ONLY=true and root not complete, evaluating latest checkpoint: ${BOLD}$latest_ckpt${NC}"
                ALL_MODELS_TO_EVALUATE+=("$latest_ckpt")
                ALL_MODEL_NAMES_FOR_EVAL+=("$model_name")
                IS_LAST_CKPT_FOR_EVAL+=("true") # Mark this single checkpoint as the 'last' one to eval for this key
                 found_evaluable_version=true
            elif [ "$LAST_STEP_ONLY" != "true" ]; then
                # Evaluate all found checkpoints if not LAST_STEP_ONLY
                for ckpt in "${CHECKPOINTS[@]}"; do
                    echo -e "${CYAN}  Adding checkpoint: ${BOLD}$ckpt${NC}"
                    ALL_MODELS_TO_EVALUATE+=("$ckpt")
                    ALL_MODEL_NAMES_FOR_EVAL+=("$model_name")
                    IS_LAST_CKPT_FOR_EVAL+=("false") # Mark as intermediate checkpoint
                    found_evaluable_version=true
                done
            fi
        else
             echo -e "${YELLOW}  No checkpoints found in $ckpt_dir${NC}"
        fi
    else
         echo -e "${YELLOW}  Checkpoint directory not found: $ckpt_dir${NC}"
    fi

    if [ "$found_evaluable_version" = false ]; then
         echo -e "${RED}  No evaluable version (completed root or checkpoints) found for model '$model_name'.${NC}"
    fi
done

TOTAL_EVAL_TASKS=${#ALL_MODELS_TO_EVALUATE[@]}
echo -e "\n${GREEN}Collected ${BOLD}$TOTAL_EVAL_TASKS${NC} total evaluation tasks.${NC}"

if [ $TOTAL_EVAL_TASKS -eq 0 ]; then
    echo -e "${YELLOW}No models or checkpoints found to evaluate based on the criteria. Exiting.${NC}"
    exit 0
fi

# --- Parallel Execution ---
# Manages running evaluation tasks across available GPUs
run_parallel_evaluation_manager() {
    local datasets_for_this_batch=$1
    echo -e "\n${BLUE}Starting parallel evaluation for datasets: ${BOLD}$datasets_for_this_batch${NC} on $NUM_GPUS GPUs${NC}"

    local pids_file=$(mktemp) # File to track background process IDs
    local status_files_prefix=$(mktemp -d)/status_ # Directory for status files

    declare -a gpu_pids # Array to store PID running on each GPU index
    declare -A pid_to_gpu_idx # Map PID back to GPU index
    for ((i=0; i<$NUM_GPUS; i++)); do
        gpu_pids[i]=-1 # Initialize GPU slots as free (-1)
    done

    local task_idx=0
    local completed_tasks=0
    local failed_tasks=0

    # Loop while there are tasks to run or tasks still running
    while [ $task_idx -lt $TOTAL_EVAL_TASKS ] || ! [[ " ${gpu_pids[@]} " =~ " -1 " ]]; do

        # Check for completed processes
        for ((gpu_idx=0; gpu_idx<$NUM_GPUS; gpu_idx++)); do
            local current_pid=${gpu_pids[gpu_idx]}
            if [ $current_pid -ne -1 ]; then
                # Check if the process associated with PID is still running
                if ! kill -0 $current_pid 2>/dev/null; then
                     echo -e "${CYAN}[Manager] Process $current_pid on GPU ${GPU_ARRAY[$gpu_idx]} finished.${NC}"
                     # Retrieve exit status (implement robustly if needed)
                     wait $current_pid
                     local exit_status=$?
                      status_file="${status_files_prefix}${current_pid}"
                      if [ -f "$status_file" ]; then
                           task_exit_status=$(cat "$status_file")
                           rm "$status_file"
                           if [ "$task_exit_status" -ne 0 ]; then
                               echo -e "${RED}[Manager] Task on GPU ${GPU_ARRAY[$gpu_idx]} (PID $current_pid) FAILED with script status $task_exit_status.${NC}"
                               failed_tasks=$((failed_tasks + 1))
                           else
                               echo -e "${GREEN}[Manager] Task on GPU ${GPU_ARRAY[$gpu_idx]} (PID $current_pid) completed successfully.${NC}"
                           fi
                      elif [ $exit_status -ne 0 ]; then
                           # Fallback if status file writing failed but wait reported error
                           echo -e "${RED}[Manager] Task on GPU ${GPU_ARRAY[$gpu_idx]} (PID $current_pid) FAILED with wait status $exit_status (status file missing).${NC}"
                           failed_tasks=$((failed_tasks + 1))
                      else
                           # If wait succeeds and no failure status file, assume success
                           echo -e "${GREEN}[Manager] Task on GPU ${GPU_ARRAY[$gpu_idx]} (PID $current_pid) completed successfully (wait status 0, no failure file).${NC}"
                      fi


                     completed_tasks=$((completed_tasks + 1))
                     gpu_pids[gpu_idx]=-1 # Mark GPU as free
                     unset pid_to_gpu_idx[$current_pid] # Remove from mapping
                fi
            fi
        done

        # Launch new tasks on free GPUs
        if [ $task_idx -lt $TOTAL_EVAL_TASKS ]; then
            for ((gpu_idx=0; gpu_idx<$NUM_GPUS; gpu_idx++)); do
                if [ ${gpu_pids[gpu_idx]} -eq -1 ]; then
                    # Found a free GPU
                    local gpu_id=${GPU_ARRAY[gpu_idx]}
                    local model_path=${ALL_MODELS_TO_EVALUATE[$task_idx]}
                    local model_name=${ALL_MODEL_NAMES_FOR_EVAL[$task_idx]}
                    local is_last=${IS_LAST_CKPT_FOR_EVAL[$task_idx]}

                    echo -e "${BLUE}[Manager] Assigning task $((task_idx + 1))/$TOTAL_EVAL_TASKS (${model_name}/$(basename $model_path)) to GPU $gpu_id${NC}"

                    # Run evaluation in the background
                    (
                         run_evaluation "$model_path" "$model_name" "$gpu_id" "$is_last" "$datasets_for_this_batch"
                         echo $? > "${status_files_prefix}$$" # Write exit code to status file named by PID
                    ) &
                    local child_pid=$!

                    gpu_pids[gpu_idx]=$child_pid
                    pid_to_gpu_idx[$child_pid]=$gpu_idx
                    echo "$child_pid" >> "$pids_file" # Track PID

                    task_idx=$((task_idx + 1)) # Move to next task

                    # Break the inner loop if all tasks assigned or no more free GPUs in this iteration
                    if [ $task_idx -ge $TOTAL_EVAL_TASKS ] || ! [[ " ${gpu_pids[@]} " =~ " -1 " ]]; then
                         break
                    fi
                fi
             done
        fi

         # If all GPUs are busy and tasks remain, wait before checking again
         if [ $task_idx -lt $TOTAL_EVAL_TASKS ] && ! [[ " ${gpu_pids[@]} " =~ " -1 " ]]; then
              echo -e "${YELLOW}[Manager] All GPUs busy. Waiting... ($completed_tasks completed, $failed_tasks failed)${NC}"
              sleep 30 # Wait for 30 seconds before checking process statuses again
         elif [ $task_idx -ge $TOTAL_EVAL_TASKS ] && ! [[ " ${gpu_pids[@]} " =~ " -1 " ]]; then
              echo -e "${YELLOW}[Manager] All tasks assigned. Waiting for ${#pid_to_gpu_idx[@]} remaining processes... ($completed_tasks completed, $failed_tasks failed)${NC}"
               sleep 30 # Wait for running processes
          elif [ $task_idx -lt $TOTAL_EVAL_TASKS ] && [[ " ${gpu_pids[@]} " =~ " -1 " ]]; then
              # This case (tasks left but GPUs free) should be handled by launching loop above
               sleep 1 # Short sleep before re-checking GPU availability
          else
               # All tasks assigned, all processes finished (gpu_pids are all -1)
               break # Exit the main while loop
           fi
    done

    echo -e "${BLUE}[Manager] All tasks for datasets [$datasets_for_this_batch] have completed processing.${NC}"
    rm "$pids_file"
    rm -rf "$(dirname ${status_files_prefix})" # Clean up status files directory

    if [ $failed_tasks -eq 0 ]; then
        echo -e "${GREEN}${BOLD}✓ All $TOTAL_EVAL_TASKS tasks completed successfully for datasets: $datasets_for_this_batch${NC}\n"
        return 0
    else
        echo -e "${RED}${BOLD}✗ $failed_tasks out of $TOTAL_EVAL_TASKS tasks failed for datasets: $datasets_for_this_batch${NC}\n"
        return 1 # Indicate failure for this batch of datasets
    fi
}


# --- Dataset Processing ---
# Handles dataset batches, installs dependencies if needed, and calls the parallel manager
process_datasets() {
    local non_olympiad_default="mathverse_testmini,mathvista_testmini_cot,mathvision_test"
    local olympiad_default="olympiadbench_test_en"
    local all_defaults="$non_olympiad_default,$olympiad_default"

    local datasets_to_process="$CUSTOM_DATASETS"
    if [ -z "$datasets_to_process" ]; then
        echo -e "${YELLOW}No custom datasets specified (--datasets). Using default dataset sets.${NC}"
        datasets_to_process="$all_defaults"
    fi

    # Separate datasets into OlympiadBench and others
    local olympiad_in_list=""
    local non_olympiad_list=""

    IFS=',' read -r -a dataset_array <<< "$datasets_to_process"
    for dataset in "${dataset_array[@]}"; do
        if [[ "$dataset" == *"olympiadbench"* ]]; then
            olympiad_in_list="$olympiad_default" # Use the exact name for processing
        else
             # Append non-olympiad dataset to the list
             if [ -z "$non_olympiad_list" ]; then
                 non_olympiad_list="$dataset"
             else
                 non_olympiad_list="$non_olympiad_list,$dataset"
             fi
        fi
    done

    local overall_success=true

    # Run non-Olympiad datasets first
    if [ -n "$non_olympiad_list" ]; then
        echo -e "${BLUE}--- Processing Non-OlympiadBench Datasets ---${NC}"
        echo "Ensuring correct dependencies for non-Olympiad math datasets..."
        pip install antlr4-python3-runtime --upgrade --quiet
        pip install latex2sympy2 --upgrade --quiet
        run_parallel_evaluation_manager "$non_olympiad_list"
        if [ $? -ne 0 ]; then
             overall_success=false
             echo -e "${RED}Failures occurred during non-OlympiadBench dataset evaluation.${NC}"
        fi
    else
        echo -e "${YELLOW}No non-OlympiadBench datasets specified or found in the list.${NC}"
    fi

    # Run OlympiadBench dataset if present
    if [ -n "$olympiad_in_list" ]; then
        echo -e "${BLUE}--- Processing OlympiadBench Dataset ---${NC}"
        echo "Ensuring correct dependencies for OlympiadBench..."
        pip install antlr4-python3-runtime==4.11 --quiet
        run_parallel_evaluation_manager "$olympiad_in_list"
         if [ $? -ne 0 ]; then
             overall_success=false
             echo -e "${RED}Failures occurred during OlympiadBench dataset evaluation.${NC}"
        fi
    else
         echo -e "${YELLOW}OlympiadBench dataset not specified or found in the list.${NC}"
    fi

    return $( $overall_success && echo 0 || echo 1 ) # Return 0 if all succeeded, 1 otherwise
}

# --- Main Execution ---
process_datasets
exit_status=$?

echo -e "\n${BLUE}============================================${NC}"
if [ $exit_status -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ All evaluation runs completed successfully.${NC}"
else
     echo -e "${RED}${BOLD}✗ Some evaluation runs failed. Please check the logs above.${NC}"
fi
echo -e "${BLUE}============================================${NC}"

exit $exit_status 