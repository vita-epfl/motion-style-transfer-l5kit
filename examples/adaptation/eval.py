# Example Evaluation

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from drivenet_eval import eval_model
# Give model path and make sure config.yaml respects the model
model_path = "examples/dro/checkpoints/vit_small_94680_steps.pth"
scene_id_to_type_path = 'dataset_metadata/validate_turns_metadata.csv'

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from stable_baselines3.common import utils
from l5kit.planning.rasterized.xmer import TransformerModel
from l5kit.planning.rasterized.model import RasterizedPlanningModel
import torch
import torch.nn as nn

dm = LocalDataManager(None)
# get config
cfg = load_config_data("examples/dro/drivenet_config.yaml")

# rasterisation and perturbation
rasterizer = build_rasterizer(cfg, dm)

# Evaluation Dataset
eval_cfg = cfg["val_data_loader"]
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
num_scenes_to_unroll = eval_cfg["max_scene_id"]

# logging
logger = utils.configure_logger(1, "./drivenet_logs/", "eval", True)

# Planning Model (e.g. Raster, Xmer)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if cfg["model_params"]["model_architecture"] in {"resnet18", "resnet50"}:
    print("CNN Model")
    model = RasterizedPlanningModel(
        model_arch=cfg["model_params"]["model_architecture"],
        num_input_channels=rasterizer.num_channels(),
        num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states
        weights_scaling=[1., 1., 1.],
        criterion=nn.MSELoss(reduction="none"),)
elif cfg["model_params"]["model_architecture"] in {"vit_tiny", "vit_small", "vit_base"}:
    print("Xmer Model")
    model = TransformerModel(
        model_arch=cfg["model_params"]["model_architecture"],
        num_input_channels=rasterizer.num_channels(),
        num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states
        weights_scaling=[1., 1., 1.],
        criterion=nn.MSELoss(reduction="none"),
        transform=cfg["model_params"]["transform"])

ckpt = torch.load(model_path, map_location=torch.device('cpu'))
if list(ckpt.keys())[0][:6] == "module":
    ckpt = {k[7:]: v for k,v in ckpt.items()}
model.load_state_dict(ckpt)
model = model.to(device)
model = model.eval()
print("Model Loaded")

# eval
import time
start = time.time()
eval_model(model, eval_dataset, logger, "eval", 2000000, num_scenes_to_unroll=4000, num_simulation_steps=None,
           enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_path)
print("Evaluation Time: ", time.time() - start)
