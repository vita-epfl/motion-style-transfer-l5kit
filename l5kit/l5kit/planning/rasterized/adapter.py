import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

class Adapter(nn.Module):
    def __init__(
        self,
        vit,
        num_memories_per_layer = 10,
        num_classes = 36,   
    ):
        super().__init__()
        # assert isinstance(vit, ViT)

        # extract some model variables needed

        dim = vit.cls_token.shape[-1]
        layers = len(vit.blocks)
        num_patches = vit.pos_embed.shape[-2]

        self.vit = vit

        # freeze ViT backbone - only memories will be finetuned

        freeze_all_layers_(vit)

        # learnable parameters

        self.memory_cls_token = nn.Parameter(torch.randn(dim))
        self.memories_per_layer = nn.Parameter(torch.randn(layers, num_memories_per_layer, dim))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # specialized attention mask to preserve the output of the original ViT
        # it allows the memory CLS token to attend to all other tokens (and the learnable memory layer tokens), but not vice versa        

        attn_mask = torch.ones((num_patches, num_patches), dtype = torch.bool)
        attn_mask = F.pad(attn_mask, (1, num_memories_per_layer), value = False)  # main tokens cannot attend to learnable memories per layer
        attn_mask = F.pad(attn_mask, (0, 0, 1, 0), value = True)                  # memory CLS token can attend to everything
        self.register_buffer('attn_mask', attn_mask)

    def img_to_tokens(self, x):
        x = self.vit.patch_embed(x)
        x = torch.cat((self.vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        return x

    def forward(self, img):
        b = img.shape[0]

        tokens = self.img_to_tokens(img)

        # add task specific memory tokens

        memory_cls_tokens = repeat(self.memory_cls_token, 'd -> b 1 d', b = b)
        tokens = torch.cat((memory_cls_tokens, tokens), dim = 1)        

        # pass memories along with image tokens through transformer for attending

        out = self.vit.forward_xmer(tokens, memories = self.memories_per_layer, attn_mask = self.attn_mask)

        # extract memory CLS tokens

        memory_cls_tokens = out[:, 0]

        # pass through task specific adapter head

        return self.mlp_head(memory_cls_tokens)
