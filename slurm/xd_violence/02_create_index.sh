#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/02_create_index_xd_violence_%A_%a.out

# Set the XD-Violence directory
xd_violence_dir="/path/to/directory/xd_violence/"

# Set paths
root_path="${xd_violence_dir}/frames"
annotationfile_path="${xd_violence_dir}/annotations/anomaly_test.txt"
batch_size=64
frame_interval=16
index_dim=1024

cap_model_names=(
    "$xd_violence_dir/captions/raw/Salesforce/blip2-opt-6.7b-coco/"
    "$xd_violence_dir/captions/raw/Salesforce/blip2-opt-6.7b/"
    "$xd_violence_dir/captions/raw/Salesforce/blip2-flan-t5-xxl/"
    "$xd_violence_dir/captions/raw/Salesforce/blip2-flan-t5-xl/"
    "$xd_violence_dir/captions/raw/Salesforce/blip2-flan-t5-xl-coco/"
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

# Activate the virtual environment
VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

index_name="index_flat_ip"
output_dir="${xd_violence_dir}/index/${names}/${index_name}/"
# shellcheck disable=SC2086 # We want to pass a list of strings
python -m src.models.create_index \
    --index_dim "$index_dim" \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_dir "${output_dir}" \
    --captions_dirs $cap_model_names_str
