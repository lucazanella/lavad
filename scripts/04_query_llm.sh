#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1

dataset_dir="YOUR_DATASET_PATH"
llm_model_name="llama-2-13b-chat"
batch_size=32
frame_interval=16
index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"  # Change this to the index name you created in scripts/02_create_index.sh

echo "Processing index: $index_name"

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

context_prompt="If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"
format_prompt="Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with  ']'."
summary_prompt="Please summarize what happened in few sentences, based on the following temporal description of a scene. Do not include any unnecessary details or descriptions."

captions_dir="$dataset_dir/captions/clean/$index_name/"

# Generate a 6-digit timestamp based on the current time
exp_id=$(date +%s | tail -c 7)

# Convert to lowercase and replace spaces with underscores
dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
# Truncate dir_name to the first 243 characters
dir_name=$(echo "$dir_name" | cut -c1-243)
dir_name=${dir_name//\//_}
# Generate a directory name based on job and task IDs and the prompt
dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")

output_scores_dir="${dataset_dir}/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"
output_summary_dir="${dataset_dir}/captions/summary/${llm_model_name}/$index_name/"

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
