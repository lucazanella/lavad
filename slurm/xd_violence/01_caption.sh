#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%5
#SBATCH --output=output/01_caption_xd_violence_%A_%a.out

# Set the XD-Violence directory
xd_violence_dir="/path/to/directory/xd_violence/"

# Set paths
root_path="${xd_violence_dir}/frames"
annotationfile_path="${xd_violence_dir}/annotations/anomaly_test.txt"
batch_size=256
frame_interval=1

# Define pretrained model names array
pretrained_model_names=(
    "Salesforce/blip2-opt-6.7b-coco"
    "Salesforce/blip2-opt-6.7b"
    "Salesforce/blip2-flan-t5-xxl"
    "Salesforce/blip2-flan-t5-xl"
    "Salesforce/blip2-flan-t5-xl-coco"
)

# Activate the virtual environment
VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Get the pretrained model name for the current task ID
pretrained_model_name="${pretrained_model_names[$SLURM_ARRAY_TASK_ID]}"
echo "Processing model: $pretrained_model_name"

output_dir="${xd_violence_dir}/captions/raw/${pretrained_model_name}/"

# Run the Python script with the specified parameters
python -m src.models.image_captioner \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --pretrained_model_name "$pretrained_model_name" \
    --output_dir "$output_dir"
