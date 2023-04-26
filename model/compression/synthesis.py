from typing import Any, Iterable, Mapping, Optional, Union

import torch
from timm.layers import to_2tuple, trunc_normal_
from torch import Tensor, nn

from misc.util import Tokenizer
from model.layers import HybridSwinStage, ResidualBlockWithStride, SwinReconLayer
from model.registry import register_synthesis_network


class SynthesisNetwork(nn.Module):
    def __init__(self):
        super(SynthesisNetwork, self).__init__()
        self.final_layer = nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def _0x2upsample2x1stride_deconv_1stage_norb_nogdn(feature_size,
                                                   bottleneck_channels,
                                                   target_dim,
                                                   output_dim_st1=96,
                                                   skip_preconv=False):
    feature_size = to_2tuple(feature_size)

    deconv_layers = [
        nn.Sequential(
            nn.Identity() if skip_preconv else ResidualBlockWithStride(in_ch=bottleneck_channels,
                                                                       out_ch=output_dim_st1,
                                                                       stride=1,
                                                                       activation=nn.LeakyReLU,
                                                                       upsample=False),
        ),
        ResidualBlockWithStride(in_ch=output_dim_st1,
                                out_ch=target_dim,
                                stride=1,
                                activation=nn.LeakyReLU,
                                upsample=False),
    ]
    embed_dims = [output_dim_st1]
    stage_input_resolutions = [(feature_size[0], feature_size[1])]
    return deconv_layers, stage_input_resolutions, embed_dims


@register_synthesis_network
class SynthesisNetworkSwinTransform(SynthesisNetwork):
    def __init__(self,
                 reconstruction_layer_config: Mapping[str, Any],
                 feature_size: int = 28,
                 bottleneck_channels: int = 48,
                 target_dim: int = 192,
                 depths: Iterable[int] = (4, 2, 2),
                 num_heads: Iterable[int] = (8, 8, 8),
                 window_sizes: Optional[Iterable[int]] = None,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[bool] = None,
                 drop_rate: int = 0,
                 stoch_drop_rate: float = 0.1,
                 attn_drop_rate: int = 0,
                 norm_layer: Optional[Union[nn.LayerNorm, nn.GroupNorm]] = nn.LayerNorm,
                 use_shortcut: bool = True):
        super(SynthesisNetworkSwinTransform, self).__init__()
        if window_sizes is None:
            window_sizes = [7 for _ in range(len(depths))]
        self.num_hybrid_stages = len(depths)
        deconv_layers, stage_input_resolutions, embed_dims = _0x2upsample2x1stride_deconv_1stage_norb_nogdn(
            feature_size=feature_size,
            bottleneck_channels=bottleneck_channels,
            target_dim=target_dim
        )
        assert len(embed_dims) == self.num_hybrid_stages

        self.deconv_embed = deconv_layers[0]
        self.tokenizer = Tokenizer()
        self.mlp_ratio = mlp_ratio

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, stoch_drop_rate, sum(depths))]

        stages = []
        self.final_resolution = to_2tuple(stage_input_resolutions[-1])

        for i_stage in range(self.num_hybrid_stages):
            stage = HybridSwinStage(dim=embed_dims[i_stage],
                                    out_dim=embed_dims[i_stage],
                                    input_resolution=stage_input_resolutions[i_stage],
                                    depth=depths[i_stage],
                                    num_heads=num_heads[i_stage],
                                    window_size=window_sizes[i_stage],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                                    attn_drop=attn_drop_rate,
                                    norm_layer=norm_layer,
                                    use_shortcut=use_shortcut,
                                    conv_module=deconv_layers[i_stage + 1])
            stages.append(stage)

        stages.append(SwinReconLayer(**reconstruction_layer_config["params"]))
        self.stages = nn.ModuleList(stages)
        self.apply(self._init_weights)

    def forward(self, x) -> Tensor:
        x = self.deconv_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_layer(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'absolute_pos_embed'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd
