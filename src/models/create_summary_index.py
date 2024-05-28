import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

from src.data.video_record import VideoRecord

sys.path.append("libs/ImageBind")
import faiss
import faiss.contrib.torch_utils
import torch
import torch.nn.functional as F
from tqdm import tqdm

from libs.ImageBind.imagebind import data
from libs.ImageBind.imagebind.models.imagebind_model import ModalityType, imagebind_huge
from src.utils.torch_utils import initialize_vlm_model_and_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dim", type=int, default=1024)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--captions_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def load_video_records(annotationfile_path: str, root_path: str) -> List[VideoRecord]:
    with open(annotationfile_path) as f:
        annotation_lines = f.readlines()
    return [VideoRecord(line.strip().split(), root_path) for line in annotation_lines]


def initialize_faiss_index(index_dim: int) -> faiss.Index:
    return faiss.IndexFlatIP(index_dim)


def filter_frame_idxs(batch_frame_idxs, frame_interval, caption_to_frame_idxs, video_captions):
    return [
        frame_idx
        for frame_idx in batch_frame_idxs
        if int(frame_idx)
        == min(
            filter(
                lambda x: x % frame_interval == 0,
                caption_to_frame_idxs[video_captions[str(frame_idx)]],
            )
        )
    ]


def add_text_to_index(
    model: torch.nn.Module, device: str, index: faiss.Index, text_list: List[str]
) -> faiss.Index:
    inputs = {ModalityType.TEXT: data.load_and_transform_text(text_list, device)}

    with torch.no_grad():
        embeddings = model(inputs)
        index_vectors = embeddings[ModalityType.TEXT].cpu().numpy()
        faiss.normalize_L2(index_vectors)
        index.add(index_vectors)

    return index


def save_results(index: faiss.Index, file_names: list, output_dir: Path, video_name: str):
    # Save faiss index
    faiss.write_index(index, str(output_dir / f"{video_name}.bin"))
    # Save file names
    with open(output_dir / f"{video_name}.json", "w") as f:
        json.dump(file_names, f)


def process_video(
    video: VideoRecord,
    model: torch.nn.Module,
    device: str,
    index_dim: int,
    batch_size: int,
    frame_interval: int,
    captions_dir: Path,
    output_dir: Path,
):
    video_name = Path(video.path).name
    index = initialize_faiss_index(index_dim)
    file_names = []

    video_caption_path = captions_dir / f"{video_name}.json"
    with open(video_caption_path) as f:
        video_captions = json.load(f)

    caption_to_frame_idxs = defaultdict(list)
    for frame_idx, caption in video_captions.items():
        caption_to_frame_idxs[caption].append(int(frame_idx))

    for batch_start_frame in tqdm(
        range(0, video.num_frames, batch_size * frame_interval),
        desc=f"Processing {video.path}",
        unit="batch",
    ):
        batch_end_frame = min(batch_start_frame + (batch_size * frame_interval), video.num_frames)
        batch_frame_idxs = range(batch_start_frame, batch_end_frame, frame_interval)

        # Only keep frames that have not been processed yet
        batch_frame_idxs = filter_frame_idxs(
            batch_frame_idxs, frame_interval, caption_to_frame_idxs, video_captions
        )

        text_list = [video_captions.get(str(frame_idx)) for frame_idx in batch_frame_idxs]

        if text_list:
            index = add_text_to_index(model, device, index, text_list)
            file_names.extend([f"{video_name}/{frame_idx}" for frame_idx in batch_frame_idxs])

    save_results(index, file_names, output_dir, video_name)


def main(
    index_dim,
    root_path,
    annotationfile_path,
    batch_size,
    frame_interval,
    captions_dir,
    output_dir,
):
    captions_dir = Path(captions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, device = initialize_vlm_model_and_device()
    video_list = load_video_records(annotationfile_path, root_path)

    for video in video_list:
        process_video(
            video,
            model,
            device,
            index_dim,
            batch_size,
            frame_interval,
            captions_dir,
            output_dir,
        )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.index_dim,
        args.root_path,
        args.annotationfile_path,
        args.batch_size,
        args.frame_interval,
        args.captions_dir,
        args.output_dir,
    )
