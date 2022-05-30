import os
import argparse
import pathlib
import random
import subprocess
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import torch
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.environment.utils import get_scene_types, get_scene_types_as_dict
from l5kit.kinematic import AckermanPerturbation
from l5kit.planning.rasterized.model import RasterizedPlanningModel
from l5kit.planning.rasterized.xmer import TransformerModel
from l5kit.planning.rasterized.xmer_adapter import TransformerAdapterModel
from l5kit.planning.rasterized.xmer_adapter2 import TransformerAdapterModel2
from l5kit.planning.rasterized.xmer_lora import TransformerLora
from l5kit.random import GaussianRandomGenerator
from l5kit.rasterization import build_rasterizer
from stable_baselines3.common import utils
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from drivenet_eval import eval_model
from utils import (subset_and_subsample, subset_and_subsample_filtered, subset_and_subsample_agents)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def load_memory_model(model, model_path):
    print("Loading Model: ", model_path)
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))

    if list(ckpt.keys())[0][:6] == "module":
        ckpt = {k[7:]: v for k,v in ckpt.items()}

    ckpt = {k[6:]: v for k,v in ckpt.items() if k[:6] == 'model.'}

    import copy
    ckpt_new = copy.deepcopy(ckpt)
    to_change = [k for k in ckpt.keys() if 'qkv' in k]
    for k in ckpt.keys():
        if k not in to_change:
            ckpt_new[k] = ckpt[k]
            continue
        if k[-6:] == 'weight':
            curr_weight = ckpt[k]
            ckpt_new[k[:-10]+'to_q.weight'] = curr_weight[:len(curr_weight)//3]
            ckpt_new[k[:-10]+'to_kv.weight'] = curr_weight[len(curr_weight)//3:]
        elif k[-4:] == 'bias':
            curr_weight = ckpt[k]
            ckpt_new[k[:-8]+'to_q.bias'] = curr_weight[:len(curr_weight)//3]
            ckpt_new[k[:-8]+'to_kv.bias'] = curr_weight[len(curr_weight)//3:]

    model.load_state_dict(ckpt_new)
    print("Model Loaded")
    return model


def load_model(model, model_path, load_strict=True):
    print("Loading Model: ", model_path)
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    if list(ckpt.keys())[0][:6] == "module":
        ckpt = {k[7:]: v for k,v in ckpt.items()}
    model.load_state_dict(ckpt, strict=load_strict)
    print("Model Loaded")
    return model

def freeze_all_but_head(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeeze head
    for _, param in model.model.head.named_parameters():
        param.requires_grad = True

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)


def freeze_all_but_LN_and_head(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeeze head
    for _, param in model.model.head.named_parameters():
        param.requires_grad = True

    # Unfreeze LayerNorm
    for name, param in model.named_parameters():
        if 'norm' in name:
            param.requires_grad = True


def unfreeze_LN(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeeze head
    for name, param in model.model.head.named_parameters():
        param.requires_grad = True

    # Unfreeze LayerNorm and Adapter
    for name, param in model.named_parameters():
        if 'norm' in name:
            param.requires_grad = True
        if 'adapter' in name:
            param.requires_grad = True


def freeze_all_but_LN_and_layer(model, layer_num):
    for _, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeeze head
    for _, param in model.model.head.named_parameters():
        param.requires_grad = True

    # Unfreeze LayerNorm
    layer_identifier = 'blocks.' + str(layer_num)
    print("Layer Identifier: ", layer_identifier)
    for name, param in model.named_parameters():
        if 'norm' in name:
            param.requires_grad = True
        if layer_identifier in name:
            param.requires_grad = True


def unfreeze_LN_and_head(model):
    # Unfreeeze head
    for _, param in model.head.named_parameters():
        param.requires_grad = True

    # Unfreeze LayerNorm
    for name, param in model.named_parameters():
        if 'norm' in name:
            param.requires_grad = True


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
cfg = load_config_data(str(path_dro / "finetune_config.yaml"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', default=1, type=float,
                        help='ratio')
    parser.add_argument('--step', default=500, type=int,
                        help='step')
    parser.add_argument('--output', default="ft_batches", type=str,
                        help='output')
    parser.add_argument('--strategy', default="all", type=str,
                        help='strategy')
    parser.add_argument('--layer_num', default=11, type=int,
                        help='Layer to finetune')
    parser.add_argument('--adapter_downsample', default=24, type=int,
                        help='Downsampling of adapters')
    parser.add_argument('--num_memory_cell', default=20, type=int,
                        help='Number of extra memory cells')
    parser.add_argument('--num_adapters', default=2, type=int,
                        help='Number of adapters, choice (1, 2, 3)')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Learning rate')
    parser.add_argument('-e', '--epochs', default=250, type=int,
                        help='Number of epochs')
    parser.add_argument('-ev', '--eval_every_n_epochs', default=25, type=int,
                        help='Eval every ev epochs')
    parser.add_argument('-s', '--seed', default=42, type=int,
                        help='Seed')
    parser.add_argument('--rank', default=8, type=int,
                        help='Rank of LoRA matrix')
    args = parser.parse_args()
    cfg["train_data_loader"]["ratio"] = args.ratio
    cfg["train_data_loader"]["step"] = args.step
    cfg["finetune_params"]["output_name"] = args.output
    cfg["finetune_params"]["strategy"] = args.strategy
    cfg["finetune_params"]["layer_num"] = args.layer_num
    cfg["finetune_params"]["adapter_downsample"] = args.adapter_downsample
    cfg["finetune_params"]["num_memory_cell"] = args.num_memory_cell
    cfg["finetune_params"]["num_adapters"] = args.num_adapters
    cfg["finetune_params"]["lr"] = args.lr
    cfg["train_data_loader"]["epochs"] = args.epochs
    cfg["train_params"]["eval_every_n_epochs"] = args.eval_every_n_epochs
    cfg["train_params"]["seed"] = args.seed
    cfg["finetune_params"]["rank"] = args.rank
    print("LR: ", cfg["finetune_params"]["lr"])

    # Get Groups (e.g. Turns, Mission)
    if cfg["train_data_loader"]["group_type"] == 'split':
        scene_id_to_type_mapping_file = str(path_l5kit / "dataset_metadata/train_split_1350.csv")
        scene_id_to_type_val_path = str(path_l5kit / "dataset_metadata/val_split_1350.csv")
    else:
        raise ValueError

    # Group Structures
    scene_type_to_id_dict = get_scene_types_as_dict(scene_id_to_type_mapping_file)
    scene_id_to_type_list = get_scene_types(scene_id_to_type_mapping_file)
    num_groups = len(scene_type_to_id_dict)
    group_counts = torch.IntTensor([len(v) for k, v in scene_type_to_id_dict.items()])
    group_str = [k for k in scene_type_to_id_dict.keys()]
    reward_scale = {"straight": 1.0, "left": 19.5, "right": 16.6}

    # Logging and Saving
    output_name = cfg["finetune_params"]["output_name"]
    if cfg["train_params"]["save_relative"]:
        save_path = path_dro / "checkpoints"
    else:
        save_path = Path("/opt/ml/checkpoints/checkpoints/")
    save_path.mkdir(parents=True, exist_ok=True)

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

    # Train Dataset
    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
    train_cfg = cfg["train_data_loader"]
    train_dataset_original = AgentDataset(cfg, train_zarr, rasterizer)
    cumulative_sizes = train_dataset_original.cumulative_sizes

    # Load train attributes
    train_cfg = cfg["train_data_loader"]
    train_scheme = train_cfg["scheme"]
    group_type = train_cfg["group_type"]
    num_epochs = train_cfg["epochs"]
    split_train = train_cfg["split"]
    # Sub-sample (for faster training)
    if not split_train:
        train_dataset = subset_and_subsample(train_dataset_original, ratio=train_cfg['ratio'], step=train_cfg['step'])
    elif train_cfg["filter_type"] in {"upper", "lower"}:
        print("Splitting Data into Upper / Lower")
        # Switch filter type when finetuning
        # filter_type = "lower" if train_cfg["filter_type"] == "upper" else "upper"
        print("Filter Type: Lower")
        # Split data into "upper" and "lower" for PETuning
        train_dataset = subset_and_subsample_filtered(train_dataset_original, ratio=train_cfg['ratio'], step=train_cfg['step'],
                                                      scene_id_to_type_list=scene_id_to_type_list,
                                                      cumulative_sizes=cumulative_sizes, filter_type="lower")
    elif train_cfg["filter_type"] in {"cycs", "cars", "peds"}:
        print("Splitting Data into Cars / Cyclists")
        print("Filter Type: ", train_cfg["filter_type"])
        train_dataset = subset_and_subsample_agents(train_dataset_original, ratio=train_cfg['ratio'], step=train_cfg['step'], filter_type=train_cfg["filter_type"])
    else:
        raise ValueError

    # Validation
    # ===== GENERATE AND LOAD CHOPPED DATASET
    num_frames_to_chop = 100
    eval_cfg = cfg["val_data_loader"]

    future_num_frames = cfg["model_params"]["future_num_frames"]
    zarr_path = Path(dm.require(eval_cfg["key"]))
    eval_base_path = zarr_path.parent / f"{zarr_path.stem}_chopped_{num_frames_to_chop}_{future_num_frames}"
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
    if filter_type in {"upper", "lower"}:
        print("Filter Type: Lower")
        # Split data into "upper" and "lower" for PETuning
        eval_dataset_adapt = subset_and_subsample_filtered(eval_dataset, ratio=1.0, step=1,
                                                           scene_id_to_type_list=scene_id_to_type_list_val,
                                                           cumulative_sizes=cumulative_sizes, filter_type="lower")
    elif filter_type in {"cars", "cycs", "peds"}:
        print("Filter Type: ", filter_type)
        # Split data into "Cars" and "Cyclists" for PETuning
        eval_dataset_adapt = subset_and_subsample_agents(eval_dataset, ratio=1.0, step=1, filter_type=train_cfg["filter_type"])

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
        if cfg["finetune_params"]["strategy"] == 'memory':
            print("Learnable Memory Model")
            model = TransformerAdapterModel(
                model_arch=cfg["model_params"]["model_architecture"],
                num_input_channels=rasterizer.num_channels(),
                num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states
                weights_scaling=[1., 1., 1.],
                criterion=nn.MSELoss(reduction="none"),
                transform=cfg["model_params"]["transform"],
                num_memories_per_layer=cfg["finetune_params"]["num_memory_cell"])
        elif cfg["finetune_params"]["strategy"] == 'adapter':
            print("Adaptor Model")
            model = TransformerAdapterModel2(
                model_arch=cfg["model_params"]["model_architecture"],
                num_input_channels=rasterizer.num_channels(),
                num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states
                weights_scaling=[1., 1., 1.],
                criterion=nn.MSELoss(reduction="none"),
                transform=cfg["model_params"]["transform"],
                adapter_downsample=cfg["finetune_params"]["adapter_downsample"],
                num_adapters=cfg["finetune_params"]["num_adapters"],)
        elif cfg["finetune_params"]["strategy"] == 'lora':
            print("LoRA Model")
            model = TransformerLora(
                model_arch=cfg["model_params"]["model_architecture"],
                num_input_channels=rasterizer.num_channels(),
                num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states
                weights_scaling=[1., 1., 1.],
                criterion=nn.MSELoss(reduction="none"),
                transform=cfg["model_params"]["transform"],
                rank=cfg["finetune_params"]["rank"])
        else:
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

    model_path = cfg["finetune_params"]["model_path"]
    if cfg["finetune_params"]["strategy"] == 'memory':
        model.model.vit = load_memory_model(model.model.vit, model_path)
    elif cfg["finetune_params"]["strategy"] == 'adapter':
        model = load_model(model, model_path, load_strict=False)
    elif cfg["finetune_params"]["strategy"] == 'lora':
        model = load_model(model, model_path, load_strict=False)
    elif cfg["finetune_params"]["strategy"] == 'scratch':
        pass
    else:
        model = load_model(model, model_path)

    if cfg["finetune_params"]["strategy"] == 'head':
        print("Finetuning Head Classifier")
        freeze_all_but_head(model)
    elif cfg["finetune_params"]["strategy"] in {'norm'}:
        print("Finetuning Layer Normalization and Classifier")
        freeze_all_but_LN_and_head(model)
    elif cfg["finetune_params"]["strategy"] == 'all':
        print("Finetuning whole model")
        pass
    elif cfg["finetune_params"]["strategy"] == 'adapter':
        print("Adapter Tuning")
        print("Finetuning Layer Normalization and Classifier")
        unfreeze_LN(model)
    elif cfg["finetune_params"]["strategy"] == 'memory':
        print("Learnable Memory Tuning")
        unfreeze_LN_and_head(model.model.vit)
    elif cfg["finetune_params"]["strategy"] == 'layer':
        layer_num = cfg["finetune_params"]["layer_num"]
        print(f"Finetuning Layer {layer_num}")
        freeze_all_but_LN_and_layer(model, layer_num)
    elif cfg["finetune_params"]["strategy"] == 'scratch':
        print("Training whole model from scratch")
        pass
    elif cfg["finetune_params"]["strategy"] == 'lora':
        print("Training LoRA")
        import loralib as lora
        lora.mark_only_lora_as_trainable(model, bias='all')
        # for _, param in model.model.head.named_parameters():
            # param.requires_grad = True
        # unfreeze_LN_and_head(model.model)
    else:
        raise ValueError

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    print("Number of Params: ", count_parameters(model))

    # model = nn.DataParallel(model)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["finetune_params"]["lr"], weight_decay=train_cfg["w_decay"])
    print("LR: ", cfg["finetune_params"]["lr"])

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
        max_lr=cfg["finetune_params"]["lr"],
        pct_start=0.3)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer,
    #     gamma=0.98)

    output_name = output_name + str(len(train_dataloader))
    print(output_name)
    # Logging
    if cfg["train_params"]["log_relative"]:
        logger = utils.configure_logger(0, str(path_dro / "drivenet_logs"), output_name, True)
    else:
        logger = utils.configure_logger(0, "/opt/ml/checkpoints/drivenet_logs", output_name, True)

    # Init Eval
    start = time.time()
    print("Evaluation")
    eval_model(model, eval_dataset_adapt, eval_gt_path, eval_cfg, logger, "eval_adapt", 0)
    print("Evaluation Time: ", time.time() - start)

    # Train
    model.train()
    torch.set_grad_enabled(True)

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # exit()
    total_steps = 0
    for epoch in range(train_cfg['epochs']):
        print(epoch , "/", train_cfg['epochs'])
        for data in tqdm(train_dataloader):
        # for data in train_dataloader:
            total_steps += 1

            # Forward pass
            data = {k: v.to(device) for k, v in data.items()}
            result = model(data)
            loss = result["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            scheduler.step()
        # scheduler.step()

        # Eval
        if (epoch + 1) % cfg["train_params"]["eval_every_n_epochs"] == 0:
            print("Evaluating............................................")
            eval_model(model, eval_dataset_adapt, eval_gt_path, eval_cfg, logger, "eval_adapt", total_steps)
            model.train()

    # print("Saving model")
    # # Final Checkpoint
    # path_to_save = str(save_path / f"{output_name}_{total_steps}_steps.pth")
    # torch.save(model.state_dict(), path_to_save)
    # # torch.save(model.cpu(), path_to_save)
    # # model = model.to(device)
    # print("Saved model")


if __name__ == "__main__":
    main()