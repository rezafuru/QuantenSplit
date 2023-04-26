from torch import Tensor, nn

from model.layers import ResidualBlockWithStride
from model.registry import register_analysis_network


@register_analysis_network
class SimpleResidualAnalysisNetwork(nn.Module):
    def __init__(self,
                 target_channels: int,
                 in_ch: int = 3,
                 in_ch1: int = 64,
                 in_ch2: int = 96,
                 **kwargs):
        super(SimpleResidualAnalysisNetwork, self).__init__()
        self.rb1 = ResidualBlockWithStride(in_ch=in_ch, out_ch=in_ch1, activation=nn.ReLU, stride=2)
        self.rb2 = ResidualBlockWithStride(in_ch=in_ch1, out_ch=in_ch2, activation=nn.ReLU, stride=2)
        self.rb3 = ResidualBlockWithStride(in_ch=in_ch2, out_ch=target_channels, activation=nn.ReLU, stride=2)

    def forward(self, x) -> Tensor:
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        return x
