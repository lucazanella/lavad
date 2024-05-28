#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/06_refine_anomaly_scores_xd_violence_%A_%a.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set the XD-Violence directory
xd_violence_dir="/path/to/directory/xd_violence/"

# Set paths
root_path="${xd_violence_dir}/frames"
annotationfile_path="${xd_violence_dir}/annotations/anomaly_test.txt"
llm_model_name="llama-2-13b-chat"
batch_size=64
frame_interval=16
fps=24
T=10
N=10
num_neighbors=10

context_prompt="How would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious or potentially criminal activities?"

exp_id="2039739_001"

# Convert to lowercase and replace spaces with underscores
dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
# Truncate dir_name to the first 243 characters
dir_name=$(echo "$dir_name" | cut -c1-243)
dir_name=${dir_name//\//_}
# Generate a directory name based on job and task IDs and the prompt
dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")

# Activate the virtual environment
VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

index_name="flan-t5-xxl"
captions_dir="$xd_violence_dir/captions/summary/${llm_model_name}/$index_name/"
index_dir="$xd_violence_dir/index/summary/${llm_model_name}/$index_name/index_flat_ip/"
scores_dir="$xd_violence_dir/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"
output_scores_dir="${xd_violence_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"
output_summary_dir="${xd_violence_dir}/captions/clean_summary/${llm_model_name}/$index_name/"
output_similarity_dir="${xd_violence_dir}/similarity/clean_summary/${llm_model_name}/${index_name}/"
output_filenames_dir="${xd_violence_dir}/filenames/clean_summary/${llm_model_name}/${index_name}/"

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
