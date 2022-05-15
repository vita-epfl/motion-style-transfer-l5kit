from typing import Dict, List

import bisect
import numpy as np

from l5kit.dataset import AgentDataset
from torch.utils.data import Subset


def subset_and_subsample(dataset: AgentDataset, ratio: float, step: int) -> Subset:

    indices = dataset.agents_indices
    indices_to_use = range(0, int(ratio * len(indices)), step)
    indices_to_use = np.array(indices_to_use)
    scene_samples = np.sort(indices_to_use)
    return Subset(dataset, scene_samples)


def subset_and_subsample_filtered(dataset: AgentDataset, ratio: float, step: int,
                                  scene_id_to_type_list: Dict[int, List[str]], cumulative_sizes: np.array,
                                  filter_type: str) -> Subset:
    # Sub-sample
    indices_to_use = range(0, int(ratio * len(dataset.agents_indices)), step)

    # Loop over scenes
    filter_ids = set()
    for curr_index in indices_to_use:
        index = dataset.agents_indices[curr_index]
        frame_index = bisect.bisect_right(dataset.cumulative_sizes_agents, index)
        scene_index = bisect.bisect_right(dataset.cumulative_sizes, frame_index)
        if scene_id_to_type_list[scene_index][0] == filter_type:
            filter_ids.add(curr_index)

    scene_samples = np.array(list(filter_ids))
    return Subset(dataset, scene_samples)


def subset_and_subsample_agents(dataset: AgentDataset, ratio: float, step: int,
                                filter_type: str) -> Subset:
    
    filter_type_to_agent_type_id = {"cars": 3, "cycs": 12, "peds": 14}
    filter_agent_id = filter_type_to_agent_type_id[filter_type]
    print("Agents: ", filter_type, " ID: ", filter_agent_id)

    import time
    start = time.time()
    # Sub-sample
    indices_to_use = range(0, int(ratio * len(dataset.agents_indices)), step)
    agents = dataset.dataset.agents

    print("Iterating over: ", len(indices_to_use))
    # Loop over scenes
    filter_ids = set()
    for curr_index in indices_to_use:
        index = dataset.agents_indices[curr_index]
        # class_id = np.argmax(agents[index]['label_probabilities'])
        # if class_id == filter_agent_id:
        if agents[index]['label_probabilities'][filter_agent_id] >= 0.5:
            filter_ids.add(curr_index)

    scene_samples = np.array(list(filter_ids))
    print("Time: ", time.time() - start)
    print("total agents: ", len(scene_samples))
    return Subset(dataset, scene_samples)
