from timm.layers import to_2tuple
from torch import nn
from torchdistill.losses.single import register_single_loss


@register_single_loss
class BppLossOrig(nn.Module):
    """
    """
    def __init__(self, entropy_module_path, input_sizes, reduction='mean'):
        super().__init__()
        self.entropy_module_path = entropy_module_path
        self.reduction = reduction
        self.input_h, self.input_w = to_2tuple(input_sizes)

    def forward(self, model_io_dict, *args, **kwargs):
        entropy_module_dict = model_io_dict[self.entropy_module_path]
        _, likelihoods = entropy_module_dict['output']
        n = likelihoods.size(0)
        if self.reduction == 'sum':
            bpp = -likelihoods.log2().sum()
        elif self.reduction == 'batchmean':
            bpp = -likelihoods.log2().sum() / n
        elif self.reduction == 'mean':
            bpp = -likelihoods.log2().sum() / (n * self.input_h * self.input_w)
        else:
            raise Exception(f"Reduction: {self.reduction} does not exist")
        return bpp
