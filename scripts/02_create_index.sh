#!/bin/bash
dataset_dir="YOUR_DATASET_PATH"
batch_size=32
frame_interval=16
index_dim=1024

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"

cap_model_names=(
    "$dataset_dir/captions/raw/Salesforce/blip2-opt-6.7b-coco/"
    "$dataset_dir/captions/raw/Salesforce/blip2-opt-6.7b/"
    "$dataset_dir/captions/raw/Salesforce/blip2-flan-t5-xxl/"
    "$dataset_dir/captions/raw/Salesforce/blip2-flan-t5-xl/"
    "$dataset_dir/captions/raw/Salesforce/blip2-flan-t5-xl-coco/"
)

cap_model_names_str=$(IFS=' '; echo "${cap_model_names[*]}")

# Extract names and concatenate with "+"
names=""
IFS='/' read -ra components <<< "$cap_model_names_str"
for component in "${components[@]}"; do
    if [[ "$component" =~ ^blip2- ]]; then
        names+="${component#blip2-}+"
    fi
done

# Remove the trailing "+" if present
names=${names%+}

echo "Creating index for $names"

index_name="index_flat_ip"
output_dir="${dataset_dir}/index/${names}/${index_name}/"
# shellcheck disable=SC2086 # We want to pass a list of strings
python -m src.models.create_index \
    --index_dim "$index_dim" \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_dir "${output_dir}" \
    --captions_dirs $cap_model_names_str
