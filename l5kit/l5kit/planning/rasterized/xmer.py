import warnings
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50
import torchvision.transforms as T

from l5kit.environment import models


class TransformerModel(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
        transform: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """Initializes the planning model.

        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        # Model_arch == "vit_base_patch16_224":
        from l5kit.timm import create_model
        import torchvision.transforms as T

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.model_arch == "vit_tiny":
            self.model = create_model('vit_tiny_patch16_224', pretrained=pretrained, drop_rate=dropout, drop_path_rate=dropout).to(device)
            # self.model = create_model('vit_tiny_patch16_224', pretrained=pretrained).to(device)
            if self.num_input_channels!=3:
                self.model.patch_embed.proj = nn.Conv2d(self.num_input_channels, 192, kernel_size=(16, 16), stride=(16, 16))
            self.model.head = nn.Linear(in_features= 192 , out_features=num_targets)
        elif self.model_arch == "vit_small_32":
            self.model = create_model('vit_small_patch32_224', pretrained=pretrained, drop_rate=dropout, drop_path_rate=dropout).to(device)
            # self.model = create_model('vit_small_patch32_224', pretrained=pretrained).to(device)
            if self.num_input_channels!=3:
                self.model.patch_embed.proj = nn.Conv2d(self.num_input_channels, 384, kernel_size=(32, 32), stride=(32, 32))
            self.model.head = nn.Linear(in_features= 384 , out_features=num_targets)
        elif self.model_arch == "vit_small":
            self.model = create_model('vit_small_patch16_224', pretrained=pretrained, drop_rate=dropout, drop_path_rate=dropout).to(device)
            # self.model = create_model('vit_small_patch16_224', pretrained=pretrained).to(device)
            if self.num_input_channels!=3:
                self.model.patch_embed.proj = nn.Conv2d(self.num_input_channels, 384, kernel_size=(16, 16), stride=(16, 16))
            self.model.head = nn.Linear(in_features= 384 , out_features=num_targets)
        elif self.model_arch == "vit_base":
            self.model = create_model('vit_base_patch16_224', pretrained=pretrained, drop_rate=dropout, drop_path_rate=dropout).to(device)
            # self.model = create_model('vit_base_patch16_224', pretrained=pretrained).to(device)
            if self.num_input_channels!=3:
                self.model.patch_embed.proj = nn.Conv2d(self.num_input_channels, 768, kernel_size=(16, 16), stride=(16, 16))
            self.model.head = nn.Linear(in_features= 768 , out_features=num_targets)
        else:
            raise NotImplementedError

        # IMG_SIZE = (224, 224)
        self.transform = transform
        # NORMALIZE_MEAN = (0.5, 0.5, 0.5)
        # NORMALIZE_STD = (0.5, 0.5, 0.5)
        NORMALIZE_MEAN = tuple([0.5] * self.num_input_channels)
        NORMALIZE_STD = tuple([0.5] * self.num_input_channels)
        transforms = [
                    # T.Resize(IMG_SIZE),
                    # T.ToTensor(),
                    T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
                    ]
        self.transforms = T.Compose(transforms)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        if self.transform:
            image_batch = self.transforms(image_batch)
        # import pdb; pdb.set_trace()
        # [batch_size, num_steps * 2]
        outputs = self.model(image_batch)
        batch_size = len(data_batch["image"])

        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            # [batch_size, num_steps * 2]
            targets = (torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)).view(
                batch_size, -1
            )
            # [batch_size, num_steps]
            target_weights = (data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
                batch_size, -1
            )
            loss = torch.mean(self.criterion(outputs, targets) * target_weights)
            train_dict = {"loss": loss}
            return train_dict
        else:
            predicted = outputs.view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = predicted[:, :, 2:3]
            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            return eval_dict

    def get_last_selfattention(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            # [batch_size, channels, height, width]
            image_batch = data_batch["image"]
            if self.transform:
                image_batch = self.transforms(image_batch)
            # import pdb; pdb.set_trace()
            # [batch_size, num_steps * 2]
            self_attns = self.model.get_last_selfattention(image_batch)
            return self_attns