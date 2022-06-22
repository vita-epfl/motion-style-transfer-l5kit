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
from l5kit.dataset import EgoDataset
from l5kit.environment.utils import get_scene_types
from l5kit.kinematic import AckermanPerturbation
from l5kit.planning.rasterized.model import RasterizedPlanningModel
from l5kit.planning.rasterized.xmer import TransformerModel
from l5kit.planning.rasterized.xmer_mosa import TransformerMoSA
from l5kit.random import GaussianRandomGenerator
from l5kit.rasterization import build_rasterizer
from stable_baselines3.common import utils
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from drivenet_eval import eval_model
from utils import subset_and_subsample, subset_and_subsample_filtered

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
    parser.add_argument('--ratio', default=0.025, type=float,
                        help='ratio')
    parser.add_argument('--step', default=5, type=int,
                        help='step')
    parser.add_argument('--output', default="ft_batches", type=str,
                        help='output')
    parser.add_argument('--strategy', default="all", type=str,
                        help='strategy')
    parser.add_argument('--layer_num', default=11, type=int,
                        help='Layer to finetune')
    parser.add_argument('--perturb', default=0.0, type=float,
                        help='Perturbation of ego (as a form of augmentation)')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Learning rate')
    parser.add_argument('-e', '--epochs', default=250, type=int,
                        help='Number of epochs')
    parser.add_argument('-ev', '--eval_every_n_epochs', default=25, type=int,
                        help='Eval every ev epochs')
    parser.add_argument('-s', '--seed', default=42, type=int,
                        help='Seed')
    parser.add_argument('--rank', default=8, type=int,
                        help='Rank of MoSA matrix')
    args = parser.parse_args()
    cfg["train_data_loader"]["ratio"] = args.ratio
    cfg["train_data_loader"]["step"] = args.step
    cfg["finetune_params"]["output_name"] = args.output
    cfg["finetune_params"]["strategy"] = args.strategy
    cfg["finetune_params"]["layer_num"] = args.layer_num
    cfg["train_data_loader"]["perturb_probability"] = args.perturb
    cfg["finetune_params"]["lr"] = args.lr
    cfg["train_data_loader"]["epochs"] = args.epochs
    cfg["train_params"]["eval_every_n_epochs"] = args.eval_every_n_epochs
    cfg["train_params"]["seed"] = args.seed
    cfg["finetune_params"]["rank"] = args.rank

    # Get Groups (e.g. Turns, Mission)
    if cfg["train_data_loader"]["group_type"] == 'split':
        scene_id_to_type_mapping_file = str(path_l5kit / "dataset_metadata/train_split_1350.csv")
        scene_id_to_type_val_path = str(path_l5kit / "dataset_metadata/val_split_1350.csv")
    # Group Structures
    scene_id_to_type_list = get_scene_types(scene_id_to_type_mapping_file)

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
    num_epochs = train_cfg["epochs"]
    split_train = train_cfg["split"]
    # Sub-sample (for faster training)
    if not split_train:
        train_dataset = subset_and_subsample(train_dataset_original, ratio=train_cfg['ratio'], step=train_cfg['step'])
    else:
        print("Splitting Data")
        # Switch filter type when finetuning
        filter_type = "lower" if train_cfg["filter_type"] == "upper" else "upper"
        print("Filter Type: ", filter_type)
        # Split data into "upper" and "lower" for Adaptation
        train_dataset = subset_and_subsample_filtered(train_dataset_original, ratio=train_cfg['ratio'], step=train_cfg['step'],
                                                    scene_id_to_type_list=scene_id_to_type_list,
                                                    cumulative_sizes=cumulative_sizes, filter_type=filter_type)

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
        if cfg["finetune_params"]["strategy"] == 'mosa':
            print("MoSA Adapter")
            model = TransformerMoSA(
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
    if cfg["finetune_params"]["strategy"] == 'mosa':
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
    elif cfg["finetune_params"]["strategy"] == 'layer':
        layer_num = cfg["finetune_params"]["layer_num"]
        print(f"Finetuning Layer {layer_num}")
        freeze_all_but_LN_and_layer(model, layer_num)
    elif cfg["finetune_params"]["strategy"] == 'scratch':
        print("Training whole model from scratch")
        pass
    elif cfg["finetune_params"]["strategy"] == 'mosa':
        print("Training MoSA Adapter")
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

    output_name = output_name + str(len(train_dataloader))
    print(output_name)
    # Logging
    if cfg["train_params"]["log_relative"]:
        logger = utils.configure_logger(0, str(path_dro / "drivenet_logs"), output_name, True)
    else:
        logger = utils.configure_logger(0, "/opt/ml/checkpoints/drivenet_logs", output_name, True)

    # Init Eval
    # start = time.time()
    # eval_model(model, eval_dataset, logger, "eval", iter_num=0, num_scenes_to_unroll=4000,
    #            enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_val_path,
    #            filter_type=filter_type)
    # print("Evaluation Time: ", time.time() - start)

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

        # Eval
        if (epoch + 1) % cfg["train_params"]["eval_every_n_epochs"] == 0:
            print("Evaluating............................................")
            eval_model(model, eval_dataset, logger, "val", total_steps, num_scenes_to_unroll,
                    enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_val_path,
                    filter_type=filter_type, start_scene_id=0)
            eval_model(model, eval_dataset, logger, "test", total_steps, num_scenes_to_unroll,
                    enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_val_path,
                    filter_type=filter_type, start_scene_id=num_scenes_to_unroll)
            model.train()

    print("Saving model")
    # Final Checkpoint
    path_to_save = str(save_path / f"{output_name}_{total_steps}_steps.pth")
    torch.save(model.module.state_dict(), path_to_save)
    # torch.save(model.cpu(), path_to_save)
    # model = model.to(device)
    print("Saved model")

    # Eval (training format)
    # print("Train waala Eval")
    # eval_model(model, eval_dataset, logger, "eval", total_steps, num_scenes_to_unroll,
    #         enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_val_path)


    # print("Starting Final Evaluation")
    # # Final Eval (Eval format)
    # start = time.time()
    # eval_model(model, eval_dataset, logger, "eval", total_steps + 10, num_scenes_to_unroll=4000,
    #            enable_scene_type_aggregation=True, scene_id_to_type_path=scene_id_to_type_val_path,
    #            filter_type=filter_type)
    # print("Evaluation Time: ", time.time() - start)
    # print(" Done Done ")

if __name__ == "__main__":
    main()