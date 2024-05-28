#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/05_create_summary_index_xd_violence_%A_%a.out

# Set the XD-Violence directory
xd_violence_dir="/path/to/directory/xd_violence/"

# Set paths
root_path="${xd_violence_dir}/frames"
annotationfile_path="${xd_violence_dir}/annotations/test.txt"
llm_model_name="llama-2-13b-chat"
batch_size=64
frame_interval=16
index_dim=1024
index_name="flan-t5-xxl"

# Activate the virtual environment
VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

captions_dir="${xd_violence_dir}/captions/summary/${llm_model_name}/${index_name}/"
output_dir="${xd_violence_dir}/index/summary/${llm_model_name}/${index_name}/index_flat_ip/"
python -m src.models.create_summary_index \
    --index_dim "$index_dim" \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --captions_dir "${captions_dir}" \
    --output_dir "${output_dir}"
