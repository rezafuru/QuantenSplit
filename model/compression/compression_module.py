from typing import Dict, Optional

from compressai.models import CompressionModel
from torch import Tensor
from torch import nn

from torchdistill.common.constant import def_logger
from torchdistill.models.registry import register_model_class

from model.registry import get_analysis_network, get_synthesis_network, register_compression_module

logger = def_logger.getChild(__name__)


@register_model_class
class CompressionModule(CompressionModel):
    """
        Embedded Variational Image Compression Model

    """

    def __init__(self,
                 entropy_bottleneck_channels: int,
                 analysis_config: Dict,
                 synthesis_config: Optional[Dict] = None,
                 ):
        super().__init__(entropy_bottleneck_channels)
        # deployed on edge client (Encoder)
        self.g_a = get_analysis_network(analysis_config["name"], **analysis_config["params"])
        if not synthesis_config:
            self.g_s = nn.Identity()
        else:
            # deployed on the server
            self.g_s = get_synthesis_network(synthesis_config["name"], **synthesis_config["params"])
        self.updated = False

    def forward(self, x: Tensor, return_likelihoods: bool):
        return NotImplementedError

    def forward_train(self, x: Tensor, return_likelihoods: bool):
        return NotImplementedError

    def compress(self, *args, **kwargs):
        raise NotImplementedError

    def decompress(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, force=False):
        logger.info("Updating Bottleneck..")
        updated = super().update(force=force)
        self.updated = True
        return updated


@register_compression_module
class FactorizedPriorModule(CompressionModule):
    """
        See FrankenSplit reposotiry for other entropy models
    """
    def __init__(self,
                 entropy_bottleneck_channels,
                 analysis_config,
                 synthesis_config=None,
                 ):
        super(FactorizedPriorModule, self).__init__(entropy_bottleneck_channels,
                                                    analysis_config,
                                                    synthesis_config)

    def get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        return medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

    def forward_train(self, x, return_likelihoods=False):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        if return_likelihoods:
            return x_hat, {"y": y_likelihoods}
        else:
            return x_hat

    def forward(self, x, return_likelihoods=False):
        if self.updated and not return_likelihoods:
            y = self.g_a(x)
            y_h = self.entropy_bottleneck.dequantize(
                self.entropy_bottleneck.quantize(y, 'dequantize', self.get_means(y))
            )
            y_h = y_h.detach()
            return self.g_s(y_h)
        return self.forward_train(x, return_likelihoods)

    def compress(self, x, *args, **kwargs):
        h = self.g_a(x)
        h_comp = self.entropy_bottleneck.compress(h)
        return h_comp, h.size()[2:]

    def decompress(self, compressed_h, *args, **kwargs):
        h_comp, h_shape = compressed_h
        h_hat = self.entropy_bottleneck.decompress(h_comp, h_shape)
        x_hat = self.g_s(h_hat)
        return x_hat

    def get_encoder_modules(self):
        return [self.g_a, self.entropy_bottleneck]
