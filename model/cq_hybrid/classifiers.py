from torch import nn
from torchdistill.models.registry import register_model_class


@register_model_class
class SimpleCClassifier(nn.Module):
    def __init__(self,
                 qubits,
                 classes):
        super(SimpleCClassifier, self).__init__()
        # note: could also try replacing linear classifier with a small MLP
        self.scores = nn.Linear(in_features=qubits, out_features=classes)

    def forward(self, x):
        return self.scores(x)


@register_model_class
class SimpleQClassifier(nn.Module):
    pass
