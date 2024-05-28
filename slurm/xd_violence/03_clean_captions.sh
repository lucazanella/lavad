#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/03_clean_captions_xd_violence_%A_%a.out

# Set the XD-Violence directory
xd_violence_dir="/path/to/directory/xd_violence/"

# Set paths
root_path="${xd_violence_dir}/frames"
annotationfile_path="${xd_violence_dir}/annotations/anomaly_test.txt"
batch_size=64
frame_interval=16
fps=24
clip_duration=10
num_samples=10
num_neighbors=1

index_name="flan-t5-xxl"

echo "Processing index: $index_name"

# Activate the virtual environment
VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

captions_dir_template="$xd_violence_dir/captions/raw/Salesforce/{}/"
index_dir="$xd_violence_dir/index/${index_name}/index_flat_ip/"
output_dir="${xd_violence_dir}/captions/clean/$index_name/"
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
