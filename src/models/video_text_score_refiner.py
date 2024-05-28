import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("libs/ImageBind")
from libs.ImageBind.imagebind import data
from libs.ImageBind.imagebind.models.imagebind_model import ModalityType
from src.data.video_record import VideoRecord
from src.utils.path_utils import find_unprocessed_videos
from src.utils.sample_utils import uniform_temporal_subsample
from src.utils.torch_utils import initialize_vlm_model_and_device


class VideoTextScoreRefiner:
    def __init__(
        self,
        model,
        device,
        output_scores_dir,
        output_summary_dir,
        output_similarity_dir,
        output_filenames_dir,
        num_samples,
        num_neighbors,
        index_dir,
        captions_dir,
        scores_dir,
        clip_duration,
        fps,
        imagefile_template,
        batch_size,
        frame_interval,
    ):
        self.model = model
        self.device = device
        self.output_scores_dir = Path(output_scores_dir)
        self.output_summary_dir = Path(output_summary_dir)
        self.output_similarity_dir = Path(output_similarity_dir)
        self.output_filenames_dir = Path(output_filenames_dir)
        self.num_samples = num_samples
        self.num_neighbors = num_neighbors
        self.index_dir = index_dir
        self.captions_dir = Path(captions_dir)
        self.scores_dir = Path(scores_dir)
        self.clip_duration = clip_duration
        self.fps = fps
        self.imagefile_template = imagefile_template
        self.batch_size = batch_size
        self.frame_interval = frame_interval

    def _load_faiss_index(self, video_name):
        index_file_path = Path(self.index_dir) / f"{video_name}.bin"
        return faiss.read_index(str(index_file_path))

    def _load_file_names(self, video_name):
        file_names_file_path = Path(self.index_dir) / f"{video_name}.json"
        with open(file_names_file_path) as f:
            return json.load(f)

    def _prepare_frame_data(self, video, batch_center_frame_idxs, frames_per_clip):
        batch_clip_frame_paths = [
            [
                Path(video.path) / self.imagefile_template.format(frame_idx)
                for frame_idx in range(
                    max(clip_center_frame - frames_per_clip // 2, 0),
                    min(clip_center_frame + frames_per_clip // 2, video.num_frames),
                )
            ]
            for clip_center_frame in batch_center_frame_idxs
        ]

        batch_clip_subsample_frame_paths = [
            uniform_temporal_subsample(clip_frame_paths, self.num_samples)
            for clip_frame_paths in batch_clip_frame_paths
        ]

        return batch_clip_subsample_frame_paths

    def _load_and_transform_data(self, batch_frame_paths):
        inputs = {
            ModalityType.VISION: data.load_and_transform_video_data(
                batch_frame_paths, self.device
            ),
        }
        return inputs

    def _calculate_search_vectors(self, inputs):
        with torch.no_grad():
            embeddings = self.model(inputs)
            search_vectors = embeddings[ModalityType.VISION].cpu().numpy()
        return search_vectors

    def _retrieve_captions(
        self,
        distances,
        indices,
        batch_frame_idxs,
        file_names,
        video_captions,
        video_captions_nn,
        video_similarity_nn,
        ret_file_names_nn,
    ):
        for idx, frame_idx in enumerate(batch_frame_idxs):
            frame_captions = {}
            frame_similarity = {}
            nn_file_names = {}

            for nn_idx in range(self.num_neighbors):
                file_name = file_names[indices[idx][nn_idx]]
                nn_file_names[str(nn_idx)] = file_name
                similarity = distances[idx][nn_idx]
                ret_index = file_name.split("/")[-1]
                frame_captions[str(nn_idx)] = video_captions[ret_index]
                frame_similarity[str(nn_idx)] = similarity.item()

            ret_file_names_nn[str(frame_idx)] = nn_file_names
            video_captions_nn[str(frame_idx)] = frame_captions
            video_similarity_nn[str(frame_idx)] = frame_similarity

    def _save_results(self, video_name, video_captions_nn, video_similarity_nn, ret_file_names_nn):
        output_path = self.output_summary_dir / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_captions_nn, f, indent=4)

        output_path = self.output_similarity_dir / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_similarity_nn, f, indent=4)

        output_path = self.output_filenames_dir / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(ret_file_names_nn, f, indent=4)

    def retrieve_nn(self, video: VideoRecord):
        video_name = Path(video.path).name
        frames_per_clip = int(self.clip_duration * self.fps)
        video_captions_nn = {}
        video_similarity_nn = {}
        ret_file_names_nn = {}

        video_caption_path = Path(self.captions_dir) / f"{video_name}.json"
        with open(video_caption_path) as f:
            video_captions = json.load(f)

        index = self._load_faiss_index(video_name)
        file_names = self._load_file_names(video_name)

        for batch_start_frame in tqdm(
            range(0, video.num_frames, self.batch_size * self.frame_interval),
            desc=f"Processing {video.path}",
            unit="batch",
        ):
            batch_end_frame = min(
                batch_start_frame + (self.batch_size * self.frame_interval), video.num_frames
            )
            batch_center_frame_idxs = range(
                batch_start_frame, batch_end_frame, self.frame_interval
            )
            batch_clip_frame_paths = self._prepare_frame_data(
                video, batch_center_frame_idxs, frames_per_clip
            )

            inputs = self._load_and_transform_data(batch_clip_frame_paths)
            search_vectors = self._calculate_search_vectors(inputs)
            faiss.normalize_L2(search_vectors)

            distances, indices = index.search(search_vectors, self.num_neighbors)

            self._retrieve_captions(
                distances,
                indices,
                batch_center_frame_idxs,
                file_names,
                video_captions,
                video_captions_nn,
                video_similarity_nn,
                ret_file_names_nn,
            )

        self._save_results(video_name, video_captions_nn, video_similarity_nn, ret_file_names_nn)

    def _load_scores(self, video_name):
        scores_file_path = self.scores_dir / f"{video_name}.json"
        with open(scores_file_path) as f:
            return json.load(f)

    def _load_ret_file_names_nn(self, video_name):
        ret_file_names_nn_file_path = self.output_filenames_dir / f"{video_name}.json"
        with open(ret_file_names_nn_file_path) as f:
            return json.load(f)

    def _save_scores(self, video_name, video_scores_nn):
        output_path = self.output_scores_dir / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_scores_nn, f, indent=4)

    def refine_scores(self, video: VideoRecord):
        video_name = Path(video.path).name
        video_scores_nn = {}

        video_scores = self._load_scores(video_name)
        ret_file_names_nn = self._load_ret_file_names_nn(video_name)

        for batch_start_frame in tqdm(
            range(0, video.num_frames, self.batch_size * self.frame_interval),
            desc=f"Processing {video.path}",
            unit="batch",
        ):
            batch_end_frame = min(
                batch_start_frame + (self.batch_size * self.frame_interval), video.num_frames
            )
            batch_center_frame_idxs = range(
                batch_start_frame, batch_end_frame, self.frame_interval
            )

            for idx, frame_idx in enumerate(batch_center_frame_idxs):
                frame_scores = {}
                for nn_idx in range(self.num_neighbors):
                    file_name = ret_file_names_nn[str(frame_idx)][str(nn_idx)]
                    ret_index = file_name.split("/")[-1]
                    frame_scores[str(nn_idx)] = video_scores[ret_index]
                video_scores_nn[str(frame_idx)] = frame_scores

        self._save_scores(video_name, video_scores_nn)


def run(
    root_path,
    annotationfile_path,
    batch_size,
    frame_interval,
    output_scores_dir,
    output_summary_dir,
    output_similarity_dir,
    output_filenames_dir,
    captions_dir,
    index_dir,
    scores_dir,
    resume,
    pathname,
    imagefile_template,
    fps,
    clip_duration,
    num_samples,
    num_neighbors,
    num_jobs,
    job_id,
):
    model, device = initialize_vlm_model_and_device()

    video_text_score_refiner = VideoTextScoreRefiner(
        model,
        device,
        output_scores_dir,
        output_summary_dir,
        output_similarity_dir,
        output_filenames_dir,
        num_samples,
        num_neighbors,
        index_dir,
        captions_dir,
        scores_dir,
        clip_duration,
        fps,
        imagefile_template,
        batch_size,
        frame_interval,
    )

    video_list = [VideoRecord(x.strip().split(), root_path) for x in open(annotationfile_path)]
    video_list = np.array_split(video_list, num_jobs)[job_id]
    if resume:
        video_list = find_unprocessed_videos(video_list, output_summary_dir, pathname)

    for video in video_list:
        video_text_score_refiner.retrieve_nn(video)
        video_text_score_refiner.refine_scores(video)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--output_scores_dir", type=str, required=True)
    parser.add_argument("--output_summary_dir", type=str, required=True)
    parser.add_argument("--output_similarity_dir", type=str, required=True)
    parser.add_argument("--output_filenames_dir", type=str, required=True)
    parser.add_argument("--captions_dir", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--scores_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pathname", type=str, default="*.json")
    parser.add_argument("--imagefile_template", type=str, default="{:06d}.jpg")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--clip_duration", type=float, default=10)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_neighbors", type=int, default=1)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.root_path,
        args.annotationfile_path,
        args.batch_size,
        args.frame_interval,
        args.output_scores_dir,
        args.output_summary_dir,
        args.output_similarity_dir,
        args.output_filenames_dir,
        args.captions_dir,
        args.index_dir,
        args.scores_dir,
        args.resume,
        args.pathname,
        args.imagefile_template,
        args.fps,
        args.clip_duration,
        args.num_samples,
        args.num_neighbors,
        args.num_jobs,
        args.job_index,
    )
