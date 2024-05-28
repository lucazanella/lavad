#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --array=0-0%1
#SBATCH --output=output/04_query_llm_xd_violence_%A_%a.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1

# Set the XD-Violence directory
xd_violence_dir="/path/to/directory/xd_violence/"

# Set paths
root_path="${xd_violence_dir}/frames"
annotationfile_path="${xd_violence_dir}/annotations/anomaly_test.txt"
llm_model_name="llama-2-13b-chat"
batch_size=64
frame_interval=16

context_prompt="How would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious or potentially criminal activities?"
format_prompt="Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with  ']'."
summary_prompt="Please summarize what happened in few sentences, based on the following temporal description of a scene. Do not include any unnecessary details or descriptions."

index_name="flan-t5-xxl"
echo "Processing index: $index_name"

captions_dir="$xd_violence_dir/captions/clean/$index_name/"

# Convert to lowercase and replace spaces with underscores
dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
# Truncate dir_name to the first 243 characters
dir_name=$(echo "$dir_name" | cut -c1-243)
dir_name=${dir_name//\//_}
# Generate a directory name based on job and task IDs and the prompt
dir_name=$(printf "%s_%03d_%s" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$dir_name")

# Activate the virtual environment
VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

output_scores_dir="${xd_violence_dir}/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"
output_summary_dir="${xd_violence_dir}/captions/summary/${llm_model_name}/$index_name/"

# Run the Python script with the specified parameters
torchrun \
    --nproc_per_node 2 --nnodes 1 -m src.models.llm_anomaly_scorer \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --summary_prompt "$summary_prompt" \
    --output_summary_dir "$output_summary_dir" \
    --captions_dir "$captions_dir" \
    --ckpt_dir libs/llama/llama-2-13b-chat/ \
    --tokenizer_path libs/llama/tokenizer.model

torchrun \
    --nproc_per_node 2 --nnodes 1 -m src.models.llm_anomaly_scorer \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_summary_dir "$output_summary_dir" \
    --context_prompt "$context_prompt" \
    --format_prompt "$format_prompt" \
    --output_scores_dir "$output_scores_dir" \
    --ckpt_dir libs/llama/llama-2-13b-chat/ \
    --tokenizer_path libs/llama/tokenizer.model \
    --score_summary
