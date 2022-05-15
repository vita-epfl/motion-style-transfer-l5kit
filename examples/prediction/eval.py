import os
import pathlib
import random
import subprocess
import time
from pathlib import Path
from tempfile import gettempdir

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import torch
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.environment.utils import get_scene_types, get_scene_types_as_dict
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
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
from utils import (subset_and_subsample, subset_and_subsample_filtered, subset_and_subsample_agents)


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
cfg = load_config_data(str(path_dro / "drivenet_config.yaml"))

# Get Groups (e.g. Turns, Mission)
if cfg["train_data_loader"]["group_type"] == 'turns':
    scene_id_to_type_mapping_file = str(path_l5kit / "dataset_metadata/train_turns_metadata.csv")
    scene_id_to_type_val_path = str(path_l5kit / "dataset_metadata/validate_turns_metadata.csv")
elif cfg["train_data_loader"]["group_type"] == 'missions':
    scene_id_to_type_mapping_file = str(path_l5kit / "dataset_metadata/train_missions.csv")
    scene_id_to_type_val_path = str(path_l5kit / "dataset_metadata/val_missions.csv")
elif cfg["train_data_loader"]["group_type"] == 'split':
    scene_id_to_type_mapping_file = str(path_l5kit / "dataset_metadata/train_split_1350.csv")
    scene_id_to_type_val_path = str(path_l5kit / "dataset_metadata/val_split_1350.csv")

# Group Structures
scene_type_to_id_dict = get_scene_types_as_dict(scene_id_to_type_mapping_file)
scene_id_to_type_list = get_scene_types(scene_id_to_type_mapping_file)
num_groups = len(scene_type_to_id_dict)
group_counts = torch.IntTensor([len(v) for k, v in scene_type_to_id_dict.items()])
group_str = [k for k in scene_type_to_id_dict.keys()]
reward_scale = {"straight": 1.0, "left": 19.5, "right": 16.6}

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

# rasterisation
rasterizer = build_rasterizer(cfg, dm)

# Load train attributes
train_cfg = cfg["train_data_loader"]
train_cfg = cfg["train_data_loader"]
train_scheme = train_cfg["scheme"]
group_type = train_cfg["group_type"]
num_epochs = train_cfg["epochs"]
split_train = train_cfg["split"]
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
elif cfg["model_params"]["model_architecture"] in {"vit_tiny", "vit_small", "vit_base", "vit_small_32"}:
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

# Load model if necessary
assert cfg["train_params"]["model_path"] != "None"
if cfg["train_params"]["model_path"] != "None":
    print("Loading model")
    checkpoint = torch.load(cfg["train_params"]["model_path"])
    # model.module.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # start_epoch = checkpoint['epoch']
    model.module.load_state_dict(checkpoint)

# Validation
# ===== GENERATE AND LOAD CHOPPED DATASET
num_frames_to_chop = 100
eval_cfg = cfg["val_data_loader"]

zarr_path = Path(dm.require(eval_cfg["key"]))
eval_base_path = zarr_path.parent / f"{zarr_path.stem}_chopped_{num_frames_to_chop}"
print(eval_base_path)

if not eval_base_path.is_dir():
    print("Chopped Dataset does not exist; it will be created")
    eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]), cfg["raster_params"]["filter_agents_threshold"], 
                                num_frames_to_chop, cfg["model_params"]["future_num_frames"], MIN_FUTURE_STEPS)

eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
eval_mask_path = str(Path(eval_base_path) / "mask.npz")
eval_gt_path = str(Path(eval_base_path) / "gt.csv")

eval_zarr = ChunkedDataset(eval_zarr_path).open()
eval_mask = np.load(eval_mask_path)["arr_0"]
# ===== INIT DATASET AND LOAD MASK
eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)

scene_id_to_type_list_val = get_scene_types(scene_id_to_type_val_path)
# print(eval_dataset)
print("Splitting Validation Data")
filter_type = train_cfg["filter_type"]
print("Filter Type: ", filter_type)
# Split data into "upper" and "lower" for PETuning
print("Upper")
eval_dataset_ind = subset_and_subsample_filtered(eval_dataset, ratio=1.0, step=1,
                                                 scene_id_to_type_list=scene_id_to_type_list_val,
                                                 cumulative_sizes=None, filter_type="upper")
print("Lower")
eval_dataset_ood = subset_and_subsample_filtered(eval_dataset, ratio=1.0, step=1,
                                                 scene_id_to_type_list=scene_id_to_type_list_val,
                                                 cumulative_sizes=None, filter_type="lower")
eval_dataset_car = subset_and_subsample_agents(eval_dataset, ratio=1.0, step=1, filter_type="cars")
eval_dataset_cyc = subset_and_subsample_agents(eval_dataset, ratio=1.0, step=1, filter_type="cycs")
eval_dataset_ped = subset_and_subsample_agents(eval_dataset, ratio=1.0, step=1, filter_type="peds")

total_steps = 100
print("Starting Final Evaluation")
eval_model(model, eval_dataset, eval_gt_path, eval_cfg, logger, "eval", total_steps+10)
print("Evaluating Splits............................................")
eval_model(model, eval_dataset_cyc, eval_gt_path, eval_cfg, logger, "eval_cyclists", total_steps+10)
eval_model(model, eval_dataset_ped, eval_gt_path, eval_cfg, logger, "eval_peds", total_steps+10)
eval_model(model, eval_dataset_ind, eval_gt_path, eval_cfg, logger, "eval_upper", total_steps+10)
eval_model(model, eval_dataset_ood, eval_gt_path, eval_cfg, logger, "eval_lower", total_steps+10)
eval_model(model, eval_dataset_car, eval_gt_path, eval_cfg, logger, "eval_cars", total_steps+10)

print(" Done Done ")
