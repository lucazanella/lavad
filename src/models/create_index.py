import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import faiss
import torch
from tqdm import tqdm

sys.path.append("libs/ImageBind")
from libs.ImageBind.imagebind import data
from libs.ImageBind.imagebind.models.imagebind_model import ModalityType
from src.data.video_record import VideoRecord
from src.utils.torch_utils import initialize_vlm_model_and_device

CAP_MODEL_NAMES = [
    "blip2-flan-t5-xl",
    "blip2-flan-t5-xl-coco",
    "blip2-flan-t5-xxl",
    "blip2-opt-6.7b",
    "blip2-opt-6.7b-coco",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dim", type=int, default=1024)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--captions_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def load_video_records(annotationfile_path: str, root_path: str) -> List[VideoRecord]:
    with open(annotationfile_path) as f:
        annotation_lines = f.readlines()
    return [VideoRecord(line.strip().split(), root_path) for line in annotation_lines]


def process_video(
    video: VideoRecord,
    model: torch.nn.Module,
    device: str,
    index_dim: int,
    batch_size: int,
    frame_interval: int,
    captions_dirs: List[str],
    output_dir: Path,
):
    video_name = Path(video.path).name
    index = init_faiss_index(index_dim)
    file_names = []
    video_captions = load_video_captions(captions_dirs, video_name)

    caption_to_frame_idxs = build_caption_to_frame_index(video_captions)

    for batch_start_frame in tqdm(
        range(0, video.num_frames, batch_size * frame_interval),
        desc=f"Processing {video.path}",
        unit="batch",
    ):
        batch_end_frame = min(batch_start_frame + (batch_size * frame_interval), video.num_frames)
        batch_frame_idxs = range(batch_start_frame, batch_end_frame, frame_interval)

        text_list = extract_text_list(
            video_captions, caption_to_frame_idxs, batch_frame_idxs, frame_interval
        )

        if text_list:
            index = update_faiss_index(model, device, index, text_list)
            file_names.extend(
                build_file_names(
                    video_captions, caption_to_frame_idxs, text_list, video_name, frame_interval
                )
            )

    save_results(index, file_names, output_dir, video_name)


def init_faiss_index(index_dim: int):
    index = faiss.IndexFlatIP(index_dim)
    return index


def load_video_captions(captions_dirs: List[str], video_name: str):
    video_captions = defaultdict(dict)

    for captions_dir in captions_dirs:
        captions_dir = Path(captions_dir)
        cap_model_name = captions_dir.name
        assert cap_model_name in CAP_MODEL_NAMES

        video_caption_path = captions_dir / f"{video_name}.json"
        with open(video_caption_path) as f:
            video_data = json.load(f)

        for frame_idx, caption in video_data.items():
            video_captions[frame_idx][cap_model_name] = caption

    return video_captions


def build_caption_to_frame_index(video_captions):
    caption_to_frame_idxs = defaultdict(list)
    for frame_idx, cap_model_name_to_caption in video_captions.items():
        frame_unique_captions = set(cap_model_name_to_caption.values())
        for caption in frame_unique_captions:
            caption_to_frame_idxs[caption].append(int(frame_idx))

    return caption_to_frame_idxs


def extract_text_list(video_captions, caption_to_frame_idxs, batch_frame_idxs, frame_interval):
    text_list = []
    for frame_idx in batch_frame_idxs:
        frame_unique_captions = set(video_captions[str(frame_idx)].values())
        unique_captions = [
            caption
            for caption in frame_unique_captions
            if int(frame_idx)
            == min(filter(lambda x: x % frame_interval == 0, caption_to_frame_idxs[caption]))
        ]
        text_list.extend(unique_captions)
    return text_list


def update_faiss_index(model, device, index, text_list):
    inputs = {ModalityType.TEXT: data.load_and_transform_text(text_list, device)}

    with torch.no_grad():
        embeddings = model(inputs)
        index_vectors = embeddings[ModalityType.TEXT].cpu().numpy()
        faiss.normalize_L2(index_vectors)
        index.add(index_vectors)

    return index


def build_file_names(video_captions, caption_to_frame_idxs, text_list, video_name, frame_interval):
    file_names = []
    for caption in text_list:
        frame_idx = min(filter(lambda x: x % frame_interval == 0, caption_to_frame_idxs[caption]))
        cap_model_name_idx = list(video_captions[str(frame_idx)].values()).index(caption)
        cap_model_name = list(video_captions[str(frame_idx)].keys())[cap_model_name_idx]
        file_names.append(f"{cap_model_name}/{video_name}/{frame_idx}")

    return file_names


def save_results(index, file_names, output_dir, video_name):
    # Save faiss index
    faiss.write_index(index, str(output_dir / f"{video_name}.bin"))
    # Save file names
    with open(output_dir / f"{video_name}.json", "w") as f:
        json.dump(file_names, f)


def main(
    index_dim: int,
    root_path: str,
    annotationfile_path: str,
    batch_size: int,
    frame_interval: int,
    captions_dirs: List[str],
    output_dir: str,
):
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
            captions_dirs,
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
        args.captions_dirs,
        args.output_dir,
    )
