import argparse
import csv
import os
from collections import Counter
from typing import Dict, Optional

import numpy as np
from l5kit.data import ChunkedDataset, LocalDataManager


#: Threshold in y-coordinate to divide scenes
Y_THRESH = -1350


def identify_split(zarr_dataset: ChunkedDataset,
                   max_frame_id: Optional[int] = None,
                   max_num_scenes: Optional[int] = None) -> Dict[int, str]:
    """Map each scene to its type based on turning in given zarr_dataset.

    :param zarr_dataset: the dataset
    :param max_frame_id: the maximum id of frame to categorize.
                         Train data has shorter frame lengths.
    :param max_num_scenes: the maximum number of scenes to categorize
    :return: the dict mapping the scene id to its type.
    """
    num_scenes = max_num_scenes or len(zarr_dataset.scenes)
    scenes = zarr_dataset.scenes[:num_scenes]

    split_dict: Dict[int, str] = {}
    # Loop Over Scenes
    for scene_id, scene_data in enumerate(scenes):
        frame_ids = scene_data["frame_index_interval"]
        start_frame, end_scene_frame = frame_ids[0], frame_ids[1]
        num_frames_in_scene = end_scene_frame - start_frame
        num_frames_to_categorize = max_frame_id or num_frames_in_scene
        end_frame = start_frame + num_frames_to_categorize
        frames = zarr_dataset.frames[start_frame:end_frame]

        y_coords = np.zeros(len(frames),)
        # iterate over frames
        for idx, frame in enumerate(frames):
            y_coords[idx] = frame["ego_translation"][1]

        # Determine Turn
        split_type = "upper"
        if min(y_coords) < Y_THRESH:
            split_type = "lower"

        # Update dict
        split_dict[scene_id] = split_type
    return split_dict

# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
# Please set the L5KIT_DATA_FOLDER environment variable
# os.environ["L5KIT_DATA_FOLDER"] = os.environ["HOME"] + '/level5_data/'
if "L5KIT_DATA_FOLDER" not in os.environ:
    raise KeyError("L5KIT_DATA_FOLDER environment variable not set")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='scenes/sample.zarr',
                        help='Path to L5Kit dataset to categorize')
    parser.add_argument('--output', type=str, default='sample_metadata.csv',
                        help='CSV file name for writing the metadata')
    args = parser.parse_args()

    # load dataset
    dm = LocalDataManager()
    dataset_path = dm.require(args.data_path)
    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()

    # categorize
    split_dict = identify_split(zarr_dataset)
    categories_counter = Counter(split_dict.values())
    print("The number of scenes per category:")
    print(categories_counter)

    # Write to csv
    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        for key, value in split_dict.items():
            writer.writerow([key, value])
