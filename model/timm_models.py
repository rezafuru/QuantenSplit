import types
from collections import OrderedDict
from typing import Optional

import torch
import timm

from torch import Tensor, nn
from torchdistill.common.constant import def_logger
from torchdistill.models.registry import register_model_func
import torch.nn.functional as F

logger = def_logger.getChild(__name__)


def forward_patch(self, x) -> Tensor:
    return self.g_a(x)


class _patch_pos_embed(nn.Module):
    def __init__(self, cls_token, pos_drop, no_embed_class, patch_embed, pos_embed):
        super(_patch_pos_embed, self).__init__()
        self.cls_token = cls_token
        self.pos_drop = pos_drop
        self.no_embed_class = no_embed_class
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed

    def forward(self, x) -> Tensor:
        x = self.patch_embed(x)
        if self.no_embed_class:
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)


def _assign_layer_names(timm_model: nn.Module) -> nn.Module:
    """
    """
    if hasattr(timm_model, "layers"):
        layers = timm_model.layers
    elif hasattr(timm_model, "blocks"):
        layers = timm_model.blocks
    else:
        raise Exception("Failed to assign layer names")
    named_layers = nn.Sequential(OrderedDict(
        [(f"layer_{i}", layer) for i, layer in enumerate(layers)]
    ))
    timm_model.layers = named_layers
    return timm_model


class TimmResnetWrapper(nn.Module):
    def __init__(self, orig, prune_stem=False):
        super(TimmResnetWrapper, self).__init__()
        self.drop_rate = orig.drop_rate
        self.stem = nn.Identity() if prune_stem else nn.Sequential(orig.conv1,
                                                                   orig.bn1,
                                                                   orig.act1,
                                                                   orig.maxpool)
        self.layers = nn.Sequential(orig.layer1,
                                    orig.layer2,
                                    orig.layer3,
                                    orig.layer4)
        self.global_pool = orig.global_pool
        self.fc = orig.fc

    def forward(self, x) -> Tensor:
        x = self.stem(x)
        x = self.layers(x)
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


def _load_custom_weights(timm_model, weights_path):
    logger.info(f"Loading pretrained weights from local {weights_path}")
    weights = torch.load(weights_path)
    if 'model' in weights:
        weights = weights['model']
    timm_model.load_state_dict(weights)
    return timm_model


@register_model_func
def get_timm_model(timm_model_name: str,
                   no_classes: int = 1000,
                   reset_head: bool = False,
                   pretrained: bool = True,
                   assign_layer_names: bool = False,
                   weights_path: Optional[str] = None,
                   skip_embed: bool = False,
                   split_idx: int = -1,
                   prune_tail: bool = False,
                   features_only: bool = False) -> nn.Module:
    # Peak performance after 5-6+ years of CS education right there
    logger.info(f"Creating {timm_model_name} timm model...")
    if pretrained and not weights_path:
        logger.info("Loading pretrained weights from timm")
        model = timm.create_model(timm_model_name, pretrained=True)
    elif pretrained and weights_path:
        logger.info(f"Loading pretrained weights from: {weights_path}")
        model = timm.create_model(timm_model_name, pretrained=False)
        if no_classes != 1000:
            model.reset_classifier(num_classes=no_classes)
        _load_custom_weights(timm_model=model, weights_path=weights_path)
    else:
        model = timm.create_model(timm_model_name, pretrained=False)
    # if "resnet" in timm_model_name:
    #     model = TimmResnetWrapper(model, prune_stem=skip_embed)
    if reset_head and not weights_path:
        logger.info(f"Creating new head with {no_classes} classes")
        model.reset_classifier(num_classes=no_classes)
    if split_idx != -1:
        logger.info(f"Splitting {timm_model_name} at stage {split_idx}")
        if prune_tail:
            model.layers = model.layers[:split_idx]
            model.head = nn.Identity()
            model.norm = nn.Identity()
            model.forward_head = lambda x: x
        else:
            if "convnext" in timm_model_name:
                model.stages = model.stages[split_idx:]
            else:
                model.layers = model.layers[split_idx:]
    if skip_embed:
        logger.info("Replacing patch embed with identity function")
        if "convnext" in timm_model_name:
            model.stem = nn.Identity()
        else:
            model.patch_embed = nn.Identity()
    if assign_layer_names:
        logger.info("Assigning stage names")
        model = _assign_layer_names(model)
    if features_only:
        model.forward = model.forward_features
    return model
