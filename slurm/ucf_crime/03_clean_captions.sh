#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/03_clean_captions_ucf_crime_%A_%a.out

# Set the UCF Crime directory
ucf_crime_dir="/path/to/directory/ucf_crime/"

# Set paths
root_path="${ucf_crime_dir}/frames"
annotationfile_path="${ucf_crime_dir}/annotations/test.txt"
batch_size=64
frame_interval=16
fps=30
clip_duration=10
num_samples=10
num_neighbors=1

index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"

echo "Processing index: $index_name"

# Activate the virtual environment
VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

captions_dir_template="$ucf_crime_dir/captions/raw/Salesforce/{}/"
index_dir="$ucf_crime_dir/index/${index_name}/index_flat_ip/"
output_dir="${ucf_crime_dir}/captions/clean/$index_name/"
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
