from typing import Any, Optional
from tqdm import tqdm
from tempfile import gettempdir

import numpy as np

import torch
from l5kit.cle.composite_metrics import CompositeMetricAggregator
from l5kit.cle.scene_type_agg import compute_cle_scene_type_aggregations, compute_scene_type_ade_fde
from l5kit.cle.validators import ValidationCountingAggregator
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.environment.callbacks import L5KitEvalCallback
from l5kit.environment.gym_metric_set import CLEMetricSet
from l5kit.environment.utils import get_scene_types
from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
from stable_baselines3.common.logger import Logger
# from joblib import Parallel, delayed
# from joblib.externals.loky import set_loky_pickler
from torch.utils.data import DataLoader
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points



def eval_model(model: torch.nn.Module, eval_dataset: AgentDataset,
               eval_gt_path: str, eval_cfg: Any, logger: Logger,
               d_set: str, iter_num: int):

    eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
                                num_workers=eval_cfg["num_workers"])
    # ==== EVAL LOOP
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    agent_ids = []

    progress_bar = tqdm(eval_dataloader)
    for data in progress_bar:
        # _, ouputs = forward(data, model, device, criterion)
        # data = {k: v.to(device) for k, v in data.items()}
        result = model(data)

        # convert agent coordinates into world offsets
        agents_coords = result["positions"].cpu().numpy()
        world_from_agents = data["world_from_agent"].numpy()
        centroids = data["centroid"].numpy()
        coords_offset = transform_points(agents_coords, world_from_agents) - centroids[:, None, :2]
        
        future_coords_offsets_pd.append(np.stack(coords_offset))
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())

    pred_path = f"{gettempdir()}/pred.csv"

    write_pred_csv(pred_path,
                timestamps=np.concatenate(timestamps),
                track_ids=np.concatenate(agent_ids),
                coords=np.concatenate(future_coords_offsets_pd),
                )

    metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
    for metric_name, metric_mean in metrics.items():
        print(metric_name, metric_mean)
        if metric_name == 'neg_multi_log_likelihood':
            logger.record(f'{d_set}/{metric_name}', metric_mean)
        elif metric_name == 'time_displace':
            logger.record(f'{d_set}/{metric_name}', metric_mean[-1])

    model.train()
    torch.set_grad_enabled(True)

    # Dump log so the evaluation results are printed with the correct timestep
    logger.record("time/total timesteps", iter_num, exclude="tensorboard")
    logger.dump(iter_num)
