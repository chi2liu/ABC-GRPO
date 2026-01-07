#!/bin/bash
set -ex

# Automatic checkpoint evaluation script for Avg@32
# Usage: ./eval_auto_checkpoints_avg32.sh <model_base_dir> <output_base_dir> [context_length] [gpu_devices]
# Example: ./eval_auto_checkpoints_avg32.sh ../../adaptive-new ./evals/avg32_results-4096 4096 0,1,2,3

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_base_dir> <output_base_dir> [context_length] [gpu_devices]"
    echo "Example: $0 ../../adaptive-new ./evals/avg32_results-4096 4096 0,1,2,3"
    exit 1
fi

MODEL_BASE_DIR=$1
OUTPUT_BASE_DIR=$2
CONTEXT_LENGTH=${3:-4096}  # Default to 4096
GPU_DEVICES=${4:-"0,1,2,3"}  # Default to 4 GPUs if not specified

# Validate model directory exists
if [ ! -d "${MODEL_BASE_DIR}" ]; then
    echo "Error: Model directory ${MODEL_BASE_DIR} does not exist"
    exit 1
fi

# GPU configuration
export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}

# Common parameters for Avg@32
PROMPT_TYPE="qwen25-math-cot"
SPLIT="test"
NUM_TEST_SAMPLE=-1
N_SAMPLING=64
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MIN_P=0

# Datasets for Avg@32 (typically competitive math datasets)
DATA_NAME="aime24,aime25,amc23,hmmt202502"
#DATA_NAME="hmmt202502"
# Function to evaluate a single checkpoint with Avg@32
evaluate_checkpoint_avg32() {
    local model_path=$1
    local output_dir=$2
    local checkpoint_name=$3

    echo "=========================================="
    echo "Evaluating checkpoint: ${checkpoint_name} (Avg@32)"
    echo "Model path: ${model_path}"
    echo "Output dir: ${output_dir}"
    echo "Context length: ${CONTEXT_LENGTH}"
    echo "N_sampling: ${N_SAMPLING}"
    echo "Temperature: ${TEMPERATURE}"
    echo "Top_p: ${TOP_P}"
    echo "Top_k: ${TOP_K}"
    echo "Min_p: ${MIN_P}"
    echo "=========================================="

    # Create output directory if it doesn't exist
    mkdir -p ${output_dir}

    # Check if evaluation already exists (to support resuming)
    metrics_exist=true
    for dataset in aime24 aime25 amc23; do
        metrics_file="${output_dir}/${dataset}/test_${PROMPT_TYPE}_-1_seed0_t${TEMPERATURE}_s0_e-1_metrics.json"
        if [ ! -f "${metrics_file}" ]; then
            metrics_exist=false
            break
        fi
    done

    if [ "$metrics_exist" = true ]; then
        echo "Avg@32 evaluation already exists for ${checkpoint_name}, checking completeness..."

        # Verify metrics files contain avg@32 data
        all_complete=true
        for dataset in aime24 aime25 amc23; do
            metrics_file="${output_dir}/${dataset}/test_${PROMPT_TYPE}_-1_seed0_t${TEMPERATURE}_s0_e-1_metrics.json"
            if [ -f "${metrics_file}" ]; then
                # Check if file contains avg@32 metric
                if ! grep -q '"avg@32"' "${metrics_file}" 2>/dev/null; then
                    all_complete=false
                    echo "  Missing avg@32 metric in ${dataset}"
                    break
                fi
            else
                all_complete=false
                break
            fi
        done

        if [ "$all_complete" = true ]; then
            echo "All Avg@32 evaluations complete for ${checkpoint_name}, skipping..."
            return
        else
            echo "Some Avg@32 evaluations missing for ${checkpoint_name}, re-running..."
        fi
    fi

    # Run Avg@32 evaluation
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
        --model_name_or_path ${model_path} \
        --data_name ${DATA_NAME} \
        --output_dir ${output_dir} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --max_tokens_per_call ${CONTEXT_LENGTH} \
        --seed 42 \
        --temperature ${TEMPERATURE} \
        --n_sampling ${N_SAMPLING} \
        --top_p ${TOP_P} \
        --top_k ${TOP_K} \
        --min_p ${MIN_P} \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs

    echo "Completed Avg@32 evaluation for ${checkpoint_name}"
}

# Function to extract checkpoint number from directory name
get_checkpoint_number() {
    local dir_name=$1
    echo "$dir_name" | grep -oE '[0-9]+' | head -1
}

# Function to get model type from path
get_model_type() {
    local model_path=$1
    local model_base=$(basename $(dirname "$model_path"))

    # Return the parent directory name as model type
    echo "$model_base"
}

# Discover and process all checkpoints
echo "=========================================="
echo "Scanning for checkpoints in: ${MODEL_BASE_DIR}"
echo "=========================================="

# Find all checkpoint directories and sort them numerically
checkpoint_dirs=()
checkpoint_nums=()

# Check for checkpoint-* pattern
for checkpoint_dir in ${MODEL_BASE_DIR}/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then
        checkpoint_name=$(basename "$checkpoint_dir")
        checkpoint_num=$(get_checkpoint_number "$checkpoint_name")
        checkpoint_dirs+=("$checkpoint_dir")
        checkpoint_nums+=("$checkpoint_num")
        echo "Found checkpoint: $checkpoint_name (number: $checkpoint_num)"
    fi
done

# If no checkpoints found, check if the base directory itself is a model
if [ ${#checkpoint_dirs[@]} -eq 0 ]; then
    echo "No checkpoint-* directories found."

    # Check if the base directory contains model files
    if [ -f "${MODEL_BASE_DIR}/config.json" ] || [ -f "${MODEL_BASE_DIR}/pytorch_model.bin" ] || [ -f "${MODEL_BASE_DIR}/model.safetensors" ]; then
        echo "Base directory appears to be a model, evaluating it directly..."
        model_name=$(basename "${MODEL_BASE_DIR}")
        model_type=$(get_model_type "${MODEL_BASE_DIR}")
        output_dir="${OUTPUT_BASE_DIR}/${model_type}/${model_name}"
        evaluate_checkpoint_avg32 ${MODEL_BASE_DIR} ${output_dir} ${model_name}
        exit 0
    else
        echo "Error: No checkpoints found and base directory is not a model"
        exit 1
    fi
fi

# Sort checkpoints numerically
sorted_indices=($(
    for i in "${!checkpoint_nums[@]}"; do
        echo "${checkpoint_nums[$i]} $i"
    done | sort -n | awk '{print $2}'
))

# Create output base directory
mkdir -p ${OUTPUT_BASE_DIR}

# Get model type for output organization
model_type=$(basename "${MODEL_BASE_DIR}")

# Save evaluation configuration
config_file="${OUTPUT_BASE_DIR}/eval_config_avg32.json"
cat > ${config_file} << EOF
{
    "model_base_dir": "${MODEL_BASE_DIR}",
    "model_type": "${model_type}",
    "output_base_dir": "${OUTPUT_BASE_DIR}",
    "gpu_devices": "${GPU_DEVICES}",
    "context_length": ${CONTEXT_LENGTH},
    "data_name": "${DATA_NAME}",
    "prompt_type": "${PROMPT_TYPE}",
    "n_sampling": ${N_SAMPLING},
    "temperature": ${TEMPERATURE},
    "top_p": ${TOP_P},
    "top_k": ${TOP_K},
    "min_p": ${MIN_P},
    "num_checkpoints": ${#checkpoint_dirs[@]},
    "timestamp": "$(date -Iseconds)"
}
EOF

echo "Saved evaluation configuration to: ${config_file}"
echo ""
echo "Found ${#checkpoint_dirs[@]} checkpoints to evaluate with Avg@32"
echo "=========================================="

# Process checkpoints in sorted order
for idx in "${sorted_indices[@]}"; do
    checkpoint_dir="${checkpoint_dirs[$idx]}"
    checkpoint_name=$(basename "$checkpoint_dir")
    checkpoint_num="${checkpoint_nums[$idx]}"

    echo ""
    echo "Processing checkpoint ${checkpoint_num} (${checkpoint_name})"
    echo "------------------------------------------"

    model_path="${checkpoint_dir}"
    output_dir="${OUTPUT_BASE_DIR}/${model_type}/${checkpoint_name}"

    evaluate_checkpoint_avg32 "${model_path}" "${output_dir}" "${checkpoint_name}"
done

echo ""
echo "=========================================="
echo "All Avg@32 evaluations completed!"
echo "Results saved to: ${OUTPUT_BASE_DIR}"
echo ""

# Generate summary using Python for better formatting
echo "Generating summary of Avg@32 results..."

python3 << EOF
import json
import os
from pathlib import Path

base_dir = "${OUTPUT_BASE_DIR}"
model_type = "${model_type}"
datasets = ["aime24", "aime25", "amc23"]
checkpoint_nums = [${checkpoint_nums[@]}]

print("\n" + "="*80)
print("AVG@32 EVALUATION SUMMARY")
print("="*80)
print(f"Model: ${MODEL_BASE_DIR}")
print(f"Output: ${OUTPUT_BASE_DIR}")
print(f"Context Length: ${CONTEXT_LENGTH}")
print("")

# Sort checkpoint numbers
checkpoint_nums.sort()

# Collect results
results_table = []
headers = ["Checkpoint"] + [ds.upper() for ds in datasets] + ["Average"]

for checkpoint_num in checkpoint_nums:
    checkpoint_dir = Path(base_dir) / model_type / f"checkpoint-{checkpoint_num}"

    if not checkpoint_dir.exists():
        continue

    row = [f"checkpoint-{checkpoint_num}"]
    dataset_scores = []

    for dataset in datasets:
        metrics_file = checkpoint_dir / dataset / f"test_${PROMPT_TYPE}_-1_seed0_t${TEMPERATURE}_s0_e-1_metrics.json"

        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                avg32 = data.get('avg@32', None)
                if avg32 is not None:
                    row.append(f"{avg32:.1f}%")
                    dataset_scores.append(avg32)
                else:
                    row.append("-")
            except:
                row.append("-")
        else:
            row.append("-")

    # Calculate average
    if dataset_scores:
        avg_score = sum(dataset_scores) / len(dataset_scores)
        row.append(f"{avg_score:.1f}%")
    else:
        row.append("-")

    results_table.append(row)

# Print table
if results_table:
    # Print headers
    header_line = " | ".join([f"{h:^15}" for h in headers])
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in results_table:
        print(" | ".join([f"{cell:^15}" for cell in row]))

    print("")

    # Find best checkpoint
    best_checkpoint = None
    best_avg = -1
    for row in results_table:
        try:
            avg_str = row[-1].replace('%', '')
            if avg_str != '-':
                avg_val = float(avg_str)
                if avg_val > best_avg:
                    best_avg = avg_val
                    best_checkpoint = row[0]
        except:
            pass

    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint} with average Avg@32 of {best_avg:.1f}%")
else:
    print("No results found.")

print("="*80)

# Save summary to file
summary_file = Path(base_dir) / "avg32_summary.json"
summary_data = {
    "model": "${MODEL_BASE_DIR}",
    "model_type": model_type,
    "context_length": ${CONTEXT_LENGTH},
    "checkpoints_evaluated": len(results_table),
    "results": []
}

for row in results_table:
    checkpoint_result = {
        "checkpoint": row[0],
        "aime24": row[1],
        "aime25": row[2],
        "amc23": row[3],
        "average": row[4]
    }
    summary_data["results"].append(checkpoint_result)

with open(summary_file, 'w') as f:
    json.dump(summary_data, f, indent=2)

print(f"\nSummary saved to: {summary_file}")
EOF

echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Review the Avg@32 results above"
echo "2. Compare with greedy (temperature=0) results if available"
echo "3. Use the best checkpoint for further evaluation or deployment"
echo "=========================================="
