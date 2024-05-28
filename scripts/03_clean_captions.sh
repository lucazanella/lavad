#!/bin/bash
dataset_dir="YOUR_DATASET_PATH"
batch_size=32
frame_interval=16
fps=30  # Change this to the frame rate of your videos
clip_duration=10
num_samples=10
num_neighbors=1
index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"  # Change this to the index name you created in scripts/02_create_index.sh

echo "Processing index: $index_name"

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

captions_dir_template="$dataset_dir/captions/raw/Salesforce/{}/"
index_dir="$dataset_dir/index/${index_name}/index_flat_ip/"
output_dir="${dataset_dir}/captions/clean/$index_name/"
python -m src.models.image_text_caption_cleaner \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_dir "$output_dir" \
    --captions_dir_template "${captions_dir_template}" \
    --index_dir "${index_dir}" \
    --fps "$fps" \
    --clip_duration "$clip_duration" \
    --num_samples "$num_samples" \
    --num_neighbors "$num_neighbors"
