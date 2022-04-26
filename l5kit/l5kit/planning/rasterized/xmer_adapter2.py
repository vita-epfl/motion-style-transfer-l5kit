import warnings
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50
import torchvision.transforms as T

from l5kit.environment import models
from l5kit.planning.rasterized.xmer import TransformerModel
from l5kit.planning.rasterized.adapter import Adapter
from l5kit.timm.models.vision_transformer_adapter2 import VisionTransformerAdapter2


class TransformerAdapterModel2(TransformerModel):
    """Raster-based planning model with adapter."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
        transform: bool = False,
        adapter_downsample: int = 8
    ) -> None:
        """Initializes the planning model.

        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__(model_arch, num_input_channels, num_targets, weights_scaling,
                         criterion, pretrained, transform)

        self.adapter_downsample = adapter_downsample
        if self.model_arch == "vit_tiny":
            self.model = VisionTransformerAdapter2(patch_size=16, embed_dim=192, depth=12, num_heads=3,
                                                   in_chans=self.num_input_channels, num_classes=num_targets,
                                                   adapter_downsample=self.adapter_downsample)
        else:
            raise NotImplementedError
