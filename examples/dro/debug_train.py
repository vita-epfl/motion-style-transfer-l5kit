import os
import pathlib
import random
import subprocess
import time
from pathlib import Path

# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import torch
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.environment.utils import get_scene_types, get_scene_types_as_dict
from l5kit.kinematic import AckermanPerturbation
from l5kit.planning.rasterized.model import RasterizedPlanningModel
from l5kit.planning.rasterized.xmer import TransformerModel
from l5kit.random import GaussianRandomGenerator
from l5kit.rasterization import build_rasterizer
from stable_baselines3.common import utils
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from drivenet_eval import eval_model
from dro_utils import (append_group_index, append_group_index_cluster, append_reward_scaling,
                       get_sample_weights, get_sample_weights_clusters, GroupBatchSampler, subset_and_subsample)
from group_dro_loss import LossComputer
from vrex_loss import VRexLossComputer

# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
# Please set the L5KIT_DATA_FOLDER environment variable
DEFAULT_L5KIT_DATA_FOLDER = '/tmp/datasets/l5kit_data'
if "L5KIT_DATA_FOLDER" not in os.environ:
    os.environ["L5KIT_DATA_FOLDER"] = DEFAULT_L5KIT_DATA_FOLDER
    if not os.path.exists(DEFAULT_L5KIT_DATA_FOLDER):
        # Download data
        subprocess.call(str( Path(__file__).parents[1] / 'download_data.sh'))

path_l5kit = Path(__file__).parents[2]
path_examples = Path(__file__).parents[1]
path_dro = Path(__file__).parent

dm = LocalDataManager(None)
# get config
cfg = load_config_data(str(path_dro / "debug_config.yaml"))

# Get Groups (e.g. Turns, Mission)
if cfg["train_data_loader"]["group_type"] == 'turns':
    scene_id_to_type_mapping_file = str(path_l5kit / "dataset_metadata/train_turns_metadata.csv")
    scene_id_to_type_val_path = str(path_l5kit / "dataset_metadata/validate_turns_metadata.csv")
elif cfg["train_data_loader"]["group_type"] == 'missions':
    scene_id_to_type_mapping_file = str(path_l5kit / "dataset_metadata/train_missions.csv")
    scene_id_to_type_val_path = str(path_l5kit / "dataset_metadata/val_missions.csv")

# Logging and Saving
output_name = cfg["train_params"]["output_name"]
if cfg["train_params"]["save_relative"]:
    save_path = path_dro / "checkpoints"
else:
    save_path = Path("/opt/ml/checkpoints/checkpoints/")
save_path.mkdir(parents=True, exist_ok=True)
if cfg["train_params"]["log_relative"]:
    logger = utils.configure_logger(0, str(path_dro / "drivenet_logs"), output_name, True)
else:
    logger = utils.configure_logger(0, "/opt/ml/checkpoints/drivenet_logs", output_name, True)

# Reproducibility
seed = cfg['train_params']['seed']
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# Reproducibility of Dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# rasterisation and perturbation
rasterizer = build_rasterizer(cfg, dm)
mean = np.array([0.0, 0.0, 0.0])  # lateral, longitudinal and angular
std = np.array([0.5, 1.5, np.pi / 6])
perturb_prob = cfg["train_data_loader"]["perturb_probability"]
perturbation = AckermanPerturbation(
    random_offset_generator=GaussianRandomGenerator(mean=mean, std=std), perturb_prob=perturb_prob)

# Train Dataset
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset_original = EgoDataset(cfg, train_zarr, rasterizer, perturbation)
cumulative_sizes = train_dataset_original.cumulative_sizes
# Load train attributes
train_cfg = cfg["train_data_loader"]
train_scheme = train_cfg["scheme"]
group_type = train_cfg["group_type"]
num_epochs = train_cfg["epochs"]
# Sub-sample (for faster training)
train_dataset = subset_and_subsample(train_dataset_original, ratio=train_cfg['ratio'], step=train_cfg['step'])

# Group Structures
scene_type_to_id_dict = get_scene_types_as_dict(scene_id_to_type_mapping_file)
scene_id_to_type_list = get_scene_types(scene_id_to_type_mapping_file)
num_groups = len(scene_type_to_id_dict)
group_counts = torch.IntTensor([len(v) for k, v in scene_type_to_id_dict.items()])
group_str = [k for k in scene_type_to_id_dict.keys()]
reward_scale = {"straight": 1.0, "left": 19.5, "right": 16.6}

# Validation Dataset (For evaluation)
eval_cfg = cfg["val_data_loader"]
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
num_scenes_to_unroll = eval_cfg["max_scene_id"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Planning Model (e.g. Raster, Xmer)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of Params: ", count_parameters(model))

model = nn.DataParallel(model)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=train_cfg["w_decay"])

# Train Loader & Schedular
g = torch.Generator()
g.manual_seed(seed)
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"], sampler=None, worker_init_fn=seed_worker,
                              generator=g)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    epochs=num_epochs,
    steps_per_epoch=len(train_dataloader),
    max_lr=5e-4,
    pct_start=0.3)

# # Train
# model.train()
# torch.set_grad_enabled(True)
# total_steps = 0
# for epoch in range(train_cfg['epochs']):
#     print(epoch , "/", train_cfg['epochs'])
#     for data in tqdm(train_dataloader):
#     # for data in train_dataloader:
#         total_steps += 1

#         # Forward pass
#         data = {k: v.to(device) for k, v in data.items()}
#         result = model(data)
#         loss = result["loss"]

#         # Backward pass
#         optimizer.zero_grad()
#         loss.sum().backward()
#         optimizer.step()
#         scheduler.step()

#     # Eval
#     if (epoch + 1) % cfg["train_params"]["eval_every_n_epochs"] == 0:
#         print("Evaluating............................................")
#         eval_model(model, eval_dataset, logger, "eval", total_steps, num_scenes_to_unroll,
#                    enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_val_path)
#         model.train()

#     # Checkpoint
#     if (epoch + 1) % cfg["train_params"]["checkpoint_every_n_epochs"] == 0:
#         print("Saving............................................")
#         path_to_save = str(save_path / f"{output_name}_{total_steps}_steps.pth")
#         torch.save(model.state_dict(), path_to_save)
#         # torch.save(model.cpu(), path_to_save)
#         # model = model.to(device)

# print("Saving model")
# # Final Checkpoint
# path_to_save = str(save_path / f"{output_name}_{total_steps}_steps.pth")
# torch.save(model.state_dict(), path_to_save)
# # torch.save(model.cpu(), path_to_save)
# # model = model.to(device)
# print("Saved model")

# Eval (training format)
# print("Train waala Eval")
# eval_model(model, eval_dataset, logger, "eval", 10, num_scenes_to_unroll=100,
#            enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_val_path)
# exit()

print("Starting Final Evaluation")
# Final Eval (Eval format)
start = time.time()
eval_model(model, eval_dataset, logger, "eval", 10, num_scenes_to_unroll=400,
           enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_val_path)
print("Evaluation Time: ", time.time() - start)
print(" Done Done ")
