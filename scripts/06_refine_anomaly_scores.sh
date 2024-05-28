#!/bin/bash
export OMP_NUM_THREADS=8

dataset_dir="YOUR_DATASET_PATH"
llm_model_name="llama-2-13b-chat"
batch_size=32
frame_interval=16
fps=30  # Change this to the frame rate of your videos
T=10
N=10
num_neighbors=10

exp_id="" # Change this to the experiment ID from scripts/04_query_llm.sh
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

captions_dir="$dataset_dir/captions/summary/${llm_model_name}/$index_name/"
index_dir="$dataset_dir/index/summary/${llm_model_name}/$index_name/index_flat_ip/"
scores_dir="$dataset_dir/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"
output_scores_dir="${dataset_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"
output_summary_dir="${dataset_dir}/captions/clean_summary/${llm_model_name}/$index_name/"
output_similarity_dir="${dataset_dir}/similarity/clean_summary/${llm_model_name}/${index_name}/"
output_filenames_dir="${dataset_dir}/filenames/clean_summary/${llm_model_name}/${index_name}/"

# Run the Python script with the specified parameters
python -m src.models.video_text_score_refiner \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_scores_dir "$output_scores_dir" \
    --output_summary_dir "$output_summary_dir" \
    --output_similarity_dir "$output_similarity_dir" \
    --output_filenames_dir "$output_filenames_dir" \
    --captions_dir "$captions_dir" \
    --index_dir "$index_dir" \
    --scores_dir "$scores_dir" \
    --fps "$fps" \
    --clip_duration "$T" \
    --num_samples "$N" \
    --num_neighbors "$num_neighbors"
