#!/bin/bash
# Script to run evaluations for multiple models with explicitly provided paths and optional custom names in parallel across specified GPUs.

# --- Environment Setup ---
# Activate conda environment if needed
# conda activate eval
# cd /path/to/your/lmms-eval  # Change this to your lmms-eval directory if needed

# Ensure necessary packages are installed
# pip install qwen_vl_utils --upgrade
# pip install loguru
# pip install antlr4-python3-runtime --upgrade # For non-olympiad math datasets
# pip install latex2sympy2 --upgrade # For non-olympiad math datasets

echo "Setting up environment variables..."
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1 # Keep if useful
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=WARN
export OPENAI_API_URL="http://localhost:8000/v1/chat/completions"
export BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="st-123"

# --- Default Configuration ---
LOG_DIR="./eval_results/main_results" # Define a log directory
DEFAULT_DATASETS="mathverse_testmini,mathvista_testmini_cot,mathvision_test"
DEFAULT_SYSTEM_PROMPT_PATH="./scripts/new_r1_system_prompt.txt"
DEFAULT_OVERWRITE="false"
DEFAULT_BATCH_SIZE=1000
DEFAULT_GPU_IDS=""
DEFAULT_MODEL_PATHS=""
DEFAULT_MODEL_NAMES="" # New default


# --- Argument Parsing with getopt ---
# Define short and long options
SHORT_OPTS="g:p:n:d:s:o:b:h"
LONG_OPTS="gpu_ids:,model_paths:,model_names:,datasets:,system_prompt_path:,overwrite:,batch_size:,help"

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
MODEL_PATHS_STR="$DEFAULT_MODEL_PATHS" # Comma-separated string
CUSTOM_DATASETS="$DEFAULT_DATASETS"
SYSTEM_PROMPT_PATH="$DEFAULT_SYSTEM_PROMPT_PATH"
OVERWRITE="$DEFAULT_OVERWRITE"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
MODEL_NAMES_STR="$DEFAULT_MODEL_NAMES" # New variable for names string
CUSTOM_NAMES_PROVIDED=false             # Flag to track if custom names were given

# Function to display help message
usage() {
    echo "Usage: $0 --gpu_ids <ids> --model_paths <paths> [options]"
    echo ""
    echo "Required arguments:"
    echo "  -g, --gpu_ids <ids>           Comma-separated GPU IDs (e.g., \"0,1,2,3\")"
    echo "  -p, --model_paths <paths>     Comma-separated list of full paths to model directories/checkpoints"
    echo "  -n, --model_names <names>     Comma-separated list of custom names for models (must match number of paths)"
    echo ""
    echo "Optional arguments:"
    echo "  -d, --datasets <datasets>     Comma-separated datasets (default: $DEFAULT_DATASETS)"
    echo "  -s, --system_prompt_path <path> Path to the system prompt file (default: $DEFAULT_SYSTEM_PROMPT_PATH)"
    echo "  -o, --overwrite <true|false>  Whether to overwrite existing results (default: $DEFAULT_OVERWRITE)"
    echo "  -b, --batch_size <num>        Evaluation batch size per GPU (default: $DEFAULT_BATCH_SIZE)"
    echo "  -h, --help                    Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --gpu_ids \"0,1\" --model_paths \"/path/to/model1,/path/to/model2/ckpt/step_100\" --datasets \"mathvista_testmini_cot\" --batch_size 16"
    exit 1
}


# Process parsed options
while true; do
    case "$1" in
        -g|--gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        -p|--model_paths)
            MODEL_PATHS_STR="$2"
            shift 2
            ;;
        -n|--model_names)
            MODEL_NAMES_STR="$2"
            CUSTOM_NAMES_PROVIDED=true
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
if [ -z "$MODEL_PATHS_STR" ]; then
    echo "Error: --model_paths is required." >&2
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

# Convert model paths string to array
IFS=',' read -r -a ALL_MODELS_TO_EVALUATE <<< "$MODEL_PATHS_STR"
TOTAL_EVAL_TASKS=${#ALL_MODELS_TO_EVALUATE[@]}

# Prepare model names array
declare -a ALL_MODEL_NAMES_FOR_EVAL
if $CUSTOM_NAMES_PROVIDED; then
   echo "Custom model names provided via --model_names."
   IFS=',' read -r -a ALL_MODEL_NAMES_FOR_EVAL <<< "$MODEL_NAMES_STR"
   # Validate lengths match
   if [ ${#ALL_MODELS_TO_EVALUATE[@]} -ne ${#ALL_MODEL_NAMES_FOR_EVAL[@]} ]; then
       echo -e "${RED}Error: The number of model paths (${#ALL_MODELS_TO_EVALUATE[@]}) must match the number of model names (${#ALL_MODEL_NAMES_FOR_EVAL[@]}).${NC}" >&2
       echo "Paths: ${ALL_MODELS_TO_EVALUATE[@]}" >&2
       echo "Names: ${ALL_MODEL_NAMES_FOR_EVAL[@]}" >&2
       exit 1
   fi
   echo -e "${GREEN}Number of model names matches number of paths.${NC}"
else
   echo "No custom model names provided. Using basenames of model paths."
   for model_path in "${ALL_MODELS_TO_EVALUATE[@]}"; do
       ALL_MODEL_NAMES_FOR_EVAL+=("$(basename "$model_path")")
   done
fi

# Validate model paths exist
echo "Validating provided model paths..."
valid_paths=true
for model_path in "${ALL_MODELS_TO_EVALUATE[@]}"; do
    if [ ! -d "$model_path" ]; then
        echo -e "${RED}Error: Provided model path does not exist or is not a directory: $model_path${NC}" >&2
        valid_paths=false
    fi
done
if [ "$valid_paths" = false ]; then
    exit 1
fi
echo -e "${GREEN}All $TOTAL_EVAL_TASKS provided model paths are valid.${NC}"


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
echo -e "${CYAN}Log Directory:     ${YELLOW}$LOG_DIR${NC}"
echo -e "${CYAN}GPU IDs:           ${YELLOW}$GPU_IDS${NC}"
echo -e "${CYAN}Number of GPUs:    ${YELLOW}$NUM_GPUS${NC}"
echo -e "${CYAN}Model Paths:       ${YELLOW}$MODEL_PATHS_STR${NC}"
echo -e "${CYAN}Number of Models:  ${YELLOW}$TOTAL_EVAL_TASKS${NC}"
echo -e "${CYAN}Custom Datasets:   ${YELLOW}${CUSTOM_DATASETS}${NC}" # Removed default message as it's now required or default set
echo -e "${CYAN}Model Names:       ${YELLOW}${ALL_MODEL_NAMES_FOR_EVAL[@]}${NC}" # Print the final list of names being used
echo -e "${CYAN}System Prompt:     ${YELLOW}${SYSTEM_PROMPT_PATH:-None}${NC}"
echo -e "${CYAN}Overwrite:         ${YELLOW}$OVERWRITE${NC}"
echo -e "${CYAN}Batch Size:        ${YELLOW}$BATCH_SIZE${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""


# --- Evaluation Function ---
# Runs a single evaluation task on a specific GPU
run_evaluation() {
    local model_dir=$1      # Full path to the model directory/checkpoint
    local gpu_id=$2         # GPU ID to use
    local datasets_to_run=$3 # Datasets for this specific run
    local model_name=$4     # Explicit model name passed from manager

    # Use the basename of the model directory for logging and output path
    local output_path="${LOG_DIR}/${model_name}"

    # Check if results exist and handle overwrite logic
    if [ -d "$output_path" ] && [ "$OVERWRITE" != "true" ]; then
        echo -e "${YELLOW}[GPU $gpu_id] Skipping: Output directory already exists for $model_name: $output_path${NC}"
        return 0
    elif [ -d "$output_path" ] && [ "$OVERWRITE" = "true" ]; then
        echo -e "${YELLOW}[GPU $gpu_id] Overwriting existing results in: $output_path${NC}"
        rm -rf "$output_path" # Clean overwrite
    fi

    mkdir -p "$output_path"

    echo -e "${GREEN}[GPU $gpu_id] Starting evaluation for ${BOLD}$model_name${NC}"
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
        --log_samples_suffix "${model_name}" \
        --output_path "$output_path" \
        --gen_kwargs "temperature=0,top_k=-1,top_p=1,max_new_tokens=8192"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[GPU $gpu_id] ✓ Evaluation completed successfully for $model_name${NC}"
    else
        echo -e "${RED}[GPU $gpu_id] ✗ Evaluation FAILED for $model_name (Exit Code: $exit_code)${NC}"
        # Optional: Create a failure marker file
        # touch "$output_path/EVALUATION_FAILED"
        return 1 # Propagate failure
    fi
    return 0
}


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
    while [ $task_idx -lt $TOTAL_EVAL_TASKS ] || [[ " ${gpu_pids[@]} " =~ " -1 " ]] == false; do

        # Check for completed processes
        for ((gpu_idx=0; gpu_idx<$NUM_GPUS; gpu_idx++)); do
            local current_pid=${gpu_pids[gpu_idx]}
            if [ $current_pid -ne -1 ]; then
                # Check if the process associated with PID is still running
                if ! kill -0 $current_pid 2>/dev/null; then
                     echo -e "${CYAN}[Manager] Process $current_pid on GPU ${GPU_ARRAY[$gpu_idx]} finished.${NC}"
                     # Retrieve exit status
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
                    local model_name=${ALL_MODEL_NAMES_FOR_EVAL[$task_idx]} # Get the corresponding name

                    echo -e "${BLUE}[Manager] Assigning task $((task_idx + 1))/$TOTAL_EVAL_TASKS ($model_name) to GPU $gpu_id${NC}"

                    # Run evaluation in the background
                    (
                         run_evaluation "$model_path" "$gpu_id" "$datasets_for_this_batch" "$model_name"
                         echo $? > "${status_files_prefix}$$" # Write exit code to status file named by PID
                    ) &
                    local child_pid=$!

                    gpu_pids[gpu_idx]=$child_pid
                    pid_to_gpu_idx[$child_pid]=$gpu_idx
                    echo "$child_pid" >> "$pids_file" # Track PID

                    task_idx=$((task_idx + 1)) # Move to next task

                    # Break the inner loop if all tasks assigned or no more free GPUs in this iteration
                    if [ $task_idx -ge $TOTAL_EVAL_TASKS ] || [[ " ${gpu_pids[@]} " =~ " -1 " ]] == false; then
                         break
                    fi
                fi
             done
        fi

         # If all GPUs are busy and tasks remain, wait before checking again
         if [ $task_idx -lt $TOTAL_EVAL_TASKS ] && [[ " ${gpu_pids[@]} " =~ " -1 " ]] == false; then
              echo -e "${YELLOW}[Manager] All GPUs busy. Waiting... ($completed_tasks completed, $failed_tasks failed)${NC}"
              sleep 30 # Wait for 30 seconds before checking process statuses again
         elif [ $task_idx -ge $TOTAL_EVAL_TASKS ] && [[ " ${gpu_pids[@]} " =~ " -1 " ]] == false; then
              echo -e "${YELLOW}[Manager] All tasks assigned. Waiting for ${#pid_to_gpu_idx[@]} remaining processes... ($completed_tasks completed, $failed_tasks failed)${NC}"
               sleep 30 # Wait for running processes
          elif [ $task_idx -lt $TOTAL_EVAL_TASKS ] && [[ " ${gpu_pids[@]} " =~ " -1 " ]] == true; then
              # This case (tasks left but GPUs free) should be handled by launching loop above
               sleep 1 # Short sleep before re-checking GPU availability
          else
               # All tasks assigned, all processes finished (gpu_pids are all -1)
               break # Exit the main while loop
           fi
    done

    echo -e "${BLUE}[Manager] All tasks for datasets [$datasets_for_this_batch] have completed processing.${NC}"
    # Clean up temporary files and directory
    if [ -f "$pids_file" ]; then
        rm "$pids_file"
    fi
    local status_dir=$(dirname "${status_files_prefix}")
    if [ -d "$status_dir" ]; then
        rm -rf "$status_dir"
    fi


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
    # Use default datasets if CUSTOM_DATASETS is empty
    if [ -z "$datasets_to_process" ]; then
        echo -e "${YELLOW}No custom datasets specified (--datasets). Using default dataset sets: $all_defaults${NC}"
        datasets_to_process="$all_defaults"
    fi

    # Separate datasets into OlympiadBench and others
    local olympiad_in_list=""
    local non_olympiad_list=""

    IFS=',' read -r -a dataset_array <<< "$datasets_to_process"
    for dataset in "${dataset_array[@]}"; do
        # Trim whitespace
        dataset=$(echo "$dataset" | xargs)
        if [[ -z "$dataset" ]]; then
            continue # Skip empty dataset names
        fi

        if [[ "$dataset" == *"olympiadbench"* ]]; then
             # Ensure only one instance of olympiad default is added
             if [ -z "$olympiad_in_list" ]; then
                 olympiad_in_list="$olympiad_default"
             fi
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

    # Return 0 if all succeeded, 1 otherwise
    if $overall_success; then
        return 0
    else
        return 1
    fi
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