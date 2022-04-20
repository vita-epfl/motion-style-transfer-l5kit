# Example Evaluation

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from drivenet_eval import eval_model
# Give model path and make sure config.yaml respects the model
model_path = "./checkpoints/drivenet_h0_p05_onecycle_schedule_step5_jit_r18_378720_steps.pt"
scene_id_to_type_path = '../../dataset_metadata/validate_turns_metadata.csv'

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from stable_baselines3.common import utils
import torch

dm = LocalDataManager(None)
# get config
cfg = load_config_data("./drivenet_config.yaml")

# rasterisation and perturbation
rasterizer = build_rasterizer(cfg, dm)

# Evaluation Dataset
eval_cfg = cfg["val_data_loader"]
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
num_scenes_to_unroll = eval_cfg["max_scene_id"]

# logging
logger = utils.configure_logger(1, "./drivenet_logs/", "eval", True)

# model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path).to(device)
model = model.eval()

# eval
eval_model(model, eval_dataset, logger, "eval", 2000000, num_scenes_to_unroll=16200, num_simulation_steps=None,
           enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_path)
