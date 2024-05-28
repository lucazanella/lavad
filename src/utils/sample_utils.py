from pathlib import Path
from typing import List

import numpy as np


def uniform_temporal_subsample(clip_frame_paths: List[Path], num_samples: int) -> List[Path]:
    t = len(clip_frame_paths)
    assert num_samples > 0 and t > 0

    if num_samples >= t:
        return clip_frame_paths

    # Calculate the indices for subsampling
    indices = np.linspace(0, t - 1, num_samples, dtype=int)
    indices = np.clip(indices, 0, t - 1)

    # Use the calculated indices to select frames
    subsampled_frames = [clip_frame_paths[i] for i in indices]

    return subsampled_frames
