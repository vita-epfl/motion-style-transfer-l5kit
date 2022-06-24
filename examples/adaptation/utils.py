from typing import Dict, Iterator, List, Optional

import numpy as np
import torch

from l5kit.dataset import EgoDataset
from torch.utils.data import Subset


def subset_and_subsample(dataset: EgoDataset, ratio: float, step: int) -> Subset:
    frames = dataset.dataset.frames
    frames_to_use = range(0, int(ratio * len(frames)), step)

    scene_samples = [dataset.get_frame_indices(f) for f in frames_to_use]
    scene_samples = np.concatenate(scene_samples).ravel()
    scene_samples = np.sort(scene_samples)
    return Subset(dataset, scene_samples)


def subset_and_subsample_filtered(dataset: EgoDataset, ratio: float, step: int,
                                  scene_id_to_type_list: Dict[int, List[str]], cumulative_sizes: np.array,
                                  filter_type: str) -> Subset:
    # Loop over scenes
    total_frames = cumulative_sizes[-1]
    cumulative_sizes = np.insert(cumulative_sizes, 0, 0)
    filter_frame_ids = [False] * total_frames
    for index in range(len(cumulative_sizes)-1):
        # Determine boundaries
        start_frame = cumulative_sizes[index]
        end_frame = cumulative_sizes[index+1]
        len_scene = end_frame - start_frame
        if scene_id_to_type_list[index][0] == filter_type:
            filter_frame_ids[start_frame : end_frame] = [True] * len_scene
    filtered_frames = [i for i, x in enumerate(filter_frame_ids) if x]

    # Filter according to ratio and step
    frames_to_use = range(0, int(ratio * len(filtered_frames)), step)
    scene_samples = [dataset.get_frame_indices(filtered_frames[f]) for f in frames_to_use]
    scene_samples = np.concatenate(scene_samples).ravel()
    scene_samples = np.sort(scene_samples)
    return Subset(dataset, scene_samples)
