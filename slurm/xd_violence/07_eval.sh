#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/07_eval_xd_violence_%A_%a.out

# Set the XD-Violence directory
xd_violence_dir="/path/to/directory/xd_violence/"

# Set paths
root_path="${xd_violence_dir}/frames"
annotationfile_path="${xd_violence_dir}/annotations/anomaly_test.txt"
temporal_annotation_file="${xd_violence_dir}/annotations/temporal_anomaly_annotation_for_testing_videos.txt"
llm_model_name="llama-2-13b-chat"
index_name="flan-t5-xxl"
frame_interval=16
num_neighbors=10
normal_label=4
video_fps=24

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

# Evaluate the AUC-ROC of clip-level scores assigned by the LLM after anomaly score refinement
captions_dir="${xd_violence_dir}/captions/clean_summary/${llm_model_name}/$index_name/"
scores_dir="${xd_violence_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"
similarity_dir="${xd_violence_dir}/similarity/clean_summary/${llm_model_name}/${index_name}/"
output_dir="${xd_violence_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"

python -m src.eval \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --temporal_annotation_file "$temporal_annotation_file" \
    --scores_dir "$scores_dir" \
    --similarity_dir "$similarity_dir" \
    --captions_dir "$captions_dir" \
    --output_dir "$output_dir" \
    --frame_interval "$frame_interval" \
    --normal_label "$normal_label" \
    --num_neighbors "$num_neighbors" \
    --video_fps "$video_fps"
