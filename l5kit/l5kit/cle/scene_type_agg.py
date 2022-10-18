from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple

import torch

from l5kit.cle.metric_set import L5MetricSet


def compute_cle_scene_type_aggregations(mset: L5MetricSet,
                                        scene_ids_to_scene_types: List[List[str]],
                                        list_validator_table_to_publish: List[str]) -> Dict[str, torch.Tensor]:
    """Compute the scene-type metric aggregations.

    :param mset: metric set to aggregate by scene type
    :param scene_ids_to_scene_types: list of scene type tags per scene
    :param list_validator_table_to_publish: list of validators for which we return structured dictionary of results
    :return: dict of result key "scene_type/validator_name" to scale tensor aggregation value.
    """

    # Set of scene types in the validation set.
    valid_scene_types: List[str] = \
        list(set([scene_type for scene_types in scene_ids_to_scene_types for scene_type in scene_types]))

    # Aggregate validator failures by scene type.
    validator_failed_frames = mset.aggregate_failed_frames()
    failed_scene_type_results: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    for vname, failed_frames in validator_failed_frames.items():
        # Aggregate scenes and frames by scene type.
        scene_set: DefaultDict[str, Set[int]] = defaultdict(set)
        frame_count: DefaultDict[str, int] = defaultdict(int)
        for scene_id, _ in failed_frames:
            scene_id = scene_id.item()
            for scene_type in scene_ids_to_scene_types[scene_id]:
                if scene_id not in scene_set[scene_type]:
                    scene_set[scene_type].add(scene_id)
                    frame_count[scene_type] += 1

        # Add scene aggregations.
        for scene_type in scene_set:
            scene_type_agg = len(scene_set[scene_type])
            failed_scene_type_results[scene_type, vname] = scene_type_agg

    # Aggregate pass/fail by scene type.
    scene_type_results: Dict[str, torch.Tensor] = {}
    for scene_type in valid_scene_types:
        for vname in mset.evaluation_plan.validators_dict():
            result_key = "/".join([scene_type, vname])
            scene_type_results[result_key] = failed_scene_type_results[scene_type, vname]

    return scene_type_results


def compute_scene_type_ade_fde(mset: L5MetricSet,
                               scene_ids_to_scene_types: List[List[str]]) -> Dict[str, float]:
    """Compute the scene-type metric aggregations for metrics ADE/FDE.

    :param mset: metric set to aggregate by scene type
    :param scene_ids_to_scene_types: list of scene type tags per scene
    :return: dict of result key "scene_type/validator_name" to scale tensor aggregation value.
    """
    scenes_results = mset.evaluator.metric_results()
    scene_type_results: Dict[str, List[float]] = defaultdict(list)
    scene_type_agg_results: DefaultDict[str, float] = defaultdict(float)

    for scene_id, scene_result in scenes_results.items():
        scene_types = scene_ids_to_scene_types[scene_id]
        l2_error = scene_result["displacement_error_l2"]
        ade, fde = l2_error[1:].mean().item(), l2_error[-1].item()
        # Add to dict
        for scene_type in scene_types:
            scene_type_results[f'fde/{scene_type}'].append(fde)
            scene_type_results[f'ade/{scene_type}'].append(ade)

    for key, value in scene_type_results.items():
        scene_type_agg_results[key] = sum(value) / len(value)

    worst_scene_type = max(scene_type_agg_results, key= lambda x: scene_type_agg_results[x])
    worst_scene_type = worst_scene_type.split('/')[-1]
    scene_type_agg_results['fde/worst_group'] = scene_type_agg_results[f'fde/{worst_scene_type}']
    scene_type_agg_results['ade/worst_group'] = scene_type_agg_results[f'ade/{worst_scene_type}']
    return scene_type_agg_results
