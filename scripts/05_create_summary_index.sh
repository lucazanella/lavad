#!/bin/bash
export OMP_NUM_THREADS=8

dataset_dir="YOUR_DATASET_PATH"
llm_model_name="llama-2-13b-chat"
batch_size=32
frame_interval=16
index_dim=1024
index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"  # Change this to the index name you created in scripts/02_create_index.sh

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

captions_dir="${dataset_dir}/captions/summary/${llm_model_name}/${index_name}/"
output_dir="${dataset_dir}/index/summary/${llm_model_name}/${index_name}/index_flat_ip/"
python -m src.models.create_summary_index \
    --index_dim "$index_dim" \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --captions_dir "${captions_dir}" \
    --output_dir "${output_dir}"
