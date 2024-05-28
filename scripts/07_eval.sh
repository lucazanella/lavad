#!/bin/bash
export OMP_NUM_THREADS=8

dataset_dir="YOUR_DATASET_PATH"
llm_model_name="llama-2-13b-chat"
frame_interval=16
num_neighbors=10
video_fps=30  # Change this to the frame rate of your videos

exp_id=""  # Change this to the experiment ID from scripts/04_query_llm.sh
index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"  # Change this to the index name you created in scripts/02_create_index.sh

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

context_prompt="If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"

# Convert to lowercase and replace spaces with underscores
dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
# Truncate dir_name to the first 243 characters
dir_name=$(echo "$dir_name" | cut -c1-243)
dir_name=${dir_name//\//_}
# Generate a directory name based on job and task IDs and the prompt
dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")

captions_dir="${dataset_dir}/captions/clean_summary/${llm_model_name}/$index_name/"
scores_dir="${dataset_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"
similarity_dir="${dataset_dir}/similarity/clean_summary/${llm_model_name}/${index_name}/"
output_dir="${dataset_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"

python -m src.eval \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --scores_dir "$scores_dir" \
    --similarity_dir "$similarity_dir" \
    --captions_dir "$captions_dir" \
    --output_dir "$output_dir" \
    --frame_interval "$frame_interval" \
    --num_neighbors "$num_neighbors" \
    --without_labels \
    --visualize \
    --video_fps "$video_fps"
