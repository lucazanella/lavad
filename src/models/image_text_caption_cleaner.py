import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import faiss  # make faiss available
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


class ImageTextCaptionCleaner:
    def __init__(
        self,
        model,
        device,
        output_dir,
        num_samples,
        num_neighbors,
        index_dir,
        captions_dir_template,
        clip_duration,
        fps,
        imagefile_template,
        batch_size,
        frame_interval,
    ):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.num_neighbors = num_neighbors
        self.index_dir = index_dir
        self.captions_dir_template = captions_dir_template
        self.clip_duration = clip_duration
        self.fps = fps
        self.imagefile_template = imagefile_template
        self.batch_size = batch_size
        self.frame_interval = frame_interval

    def process_video(self, video: VideoRecord):
        video_name = Path(video.path).name
        frames_per_clip = int(self.clip_duration * self.fps)
        video_captions_retrieved = defaultdict(dict)

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
            batch_frame_idxs, batch_frame_paths = self._prepare_frame_data(
                video, batch_center_frame_idxs, frames_per_clip
            )

            inputs = self._load_and_transform_data(batch_frame_paths)
            search_vectors = self._calculate_search_vectors(inputs)
            faiss.normalize_L2(search_vectors)

            distances, indices = index.search(search_vectors, self.num_neighbors)

            self._retrieve_captions(
                indices,
                batch_frame_idxs,
                file_names,
                video_name,
                batch_center_frame_idxs,
                video_captions_retrieved,
            )

        self._save_results(video_name, video_captions_retrieved)

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

        batch_frame_paths = [
            frame_path
            for clip_frame_paths in batch_clip_subsample_frame_paths
            for frame_path in clip_frame_paths
        ]

        # filenames' name is the frame index
        batch_frame_idxs = [int(Path(frame_path).stem) for frame_path in batch_frame_paths]
        return batch_frame_idxs, batch_frame_paths

    def _load_and_transform_data(self, batch_frame_paths):
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(
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
        indices,
        batch_frame_idxs,
        file_names,
        video_name,
        batch_center_frame_idxs,
        video_captions_retrieved,
    ):
        for idx, frame_idx in enumerate(batch_frame_idxs):
            file_name = file_names[indices[idx][0]]
            ret_cap_model_name, ret_video_name, ret_index = file_name.split("/")

            captions_dir = Path(self.captions_dir_template.format(ret_cap_model_name))
            video_caption_path = Path(captions_dir) / f"{video_name}.json"
            with open(video_caption_path) as f:
                video_captions = json.load(f)

            center_frame_idx = batch_center_frame_idxs[idx // self.num_samples]

            video_captions_retrieved[str(center_frame_idx)][str(frame_idx)] = video_captions[
                ret_index
            ]

    def _save_results(self, video_name, video_captions_retrieved):
        output_path = self.output_dir / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_captions_retrieved, f, indent=4)


def run(
    root_path: str,
    annotationfile_path: str,
    batch_size: int,
    frame_interval: int,
    captions_dir_template: str,
    output_dir: str,
    index_dir: str,
    imagefile_template: str,
    fps: float,
    clip_duration: float,
    num_samples: int,
    num_neighbors: int,
    num_jobs: int,
    job_id: int,
    resume: bool,
    pathname: str,
):
    model, device = initialize_vlm_model_and_device()

    image_text_caption_cleaner = ImageTextCaptionCleaner(
        model,
        device,
        output_dir,
        num_samples,
        num_neighbors,
        index_dir,
        captions_dir_template,
        clip_duration,
        fps,
        imagefile_template,
        batch_size,
        frame_interval,
    )

    video_list = [VideoRecord(x.strip().split(), root_path) for x in open(annotationfile_path)]
    video_list = np.array_split(video_list, num_jobs)[job_id]
    if resume:
        video_list = find_unprocessed_videos(video_list, output_dir, pathname)

    for video in video_list:
        image_text_caption_cleaner.process_video(video)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--captions_dir_template", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pathname", type=str, default="*.json")
    parser.add_argument("--imagefile_template", type=str, default="{:06d}.jpg")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--clip_duration", type=float, default=10)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_neighbors", type=int, default=1)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_id", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        root_path=args.root_path,
        annotationfile_path=args.annotationfile_path,
        batch_size=args.batch_size,
        frame_interval=args.frame_interval,
        output_dir=args.output_dir,
        captions_dir_template=args.captions_dir_template,
        index_dir=args.index_dir,
        imagefile_template=args.imagefile_template,
        fps=args.fps,
        clip_duration=args.clip_duration,
        num_samples=args.num_samples,
        num_neighbors=args.num_neighbors,
        num_jobs=args.num_jobs,
        job_id=args.job_id,
        resume=args.resume,
        pathname=args.pathname,
    )
