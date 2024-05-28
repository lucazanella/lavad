#!/bin/bash
dataset_dir="YOUR_DATASET_PATH"
batch_size=32
frame_interval=1

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

# Define pretrained model names array
pretrained_model_names=(
    "Salesforce/blip2-opt-6.7b-coco"
    "Salesforce/blip2-opt-6.7b"
    "Salesforce/blip2-flan-t5-xxl"
    "Salesforce/blip2-flan-t5-xl"
    "Salesforce/blip2-flan-t5-xl-coco"
)

for pretrained_model_name in "${pretrained_model_names[@]}"; do
    echo "Processing model: $pretrained_model_name"

    output_dir="${dataset_dir}/captions/raw/${pretrained_model_name}/"

    python -m src.models.image_captioner \
        --root_path "$root_path" \
        --annotationfile_path "$annotationfile_path" \
        --batch_size "$batch_size" \
        --frame_interval "$frame_interval" \
        --pretrained_model_name "$pretrained_model_name" \
        --output_dir "$output_dir"
done
