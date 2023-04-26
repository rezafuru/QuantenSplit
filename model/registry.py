from typing import Callable

from torch import nn
from torchdistill.common.constant import def_logger
from torchdistill.models.registry import register_model_class

SYNTHESIS_NETWORK_DICT = dict()
COMPRESSOR_DICT = dict()
ANALYSIS_NETWORK_DICT = dict()
HYBRID_CQNODE_DICT = dict()
HYPER_NETWORK_DICT = dict()
Q_CIRCUIT_DICT = dict()
CUSTOM_COMPRESSION_MODULE_DICT = dict()

logger = def_logger.getChild(__name__)


def register_q_circuit(fun: Callable):
    Q_CIRCUIT_DICT[fun.__name__] = fun
    return fun


def get_q_circuit(circuit_name) -> Callable:
    if circuit_name not in Q_CIRCUIT_DICT:
        raise ValueError("circuit with name `{}` not registered".format(circuit_name))
    return Q_CIRCUIT_DICT[circuit_name]


def register_hybrid_cqnode(cls):
    HYBRID_CQNODE_DICT[cls.__name__] = cls
    return cls


def get_hybrid_cqnode(node_name, **kwargs):
    if node_name not in HYBRID_CQNODE_DICT:
        raise ValueError("hybrid cq node with name `{}` not registered".format(node_name))
    return HYBRID_CQNODE_DICT[node_name](**kwargs)


def register_compression_module(cls):
    COMPRESSOR_DICT[cls.__name__] = cls
    return cls


def get_compression_module(compressor_name, **kwargs):
    if compressor_name not in COMPRESSOR_DICT:
        raise ValueError("compressor with name `{}` not registered".format(compressor_name))
    return COMPRESSOR_DICT[compressor_name](**kwargs)


def register_synthesis_network(cls):
    SYNTHESIS_NETWORK_DICT[cls.__name__] = cls
    return cls


def get_synthesis_network(synthesis_network_name, **kwargs):
    if synthesis_network_name not in SYNTHESIS_NETWORK_DICT:
        raise ValueError("synthesis network with name `{}` not registered".format(synthesis_network_name))
    return SYNTHESIS_NETWORK_DICT[synthesis_network_name](**kwargs)


def register_analysis_network(cls):
    ANALYSIS_NETWORK_DICT[cls.__name__] = cls
    return cls


def get_analysis_network(analysis_network_name, **kwargs):
    if analysis_network_name not in ANALYSIS_NETWORK_DICT:
        raise ValueError("analysis network with name `{}` not registered".format(analysis_network_name))

    return ANALYSIS_NETWORK_DICT[analysis_network_name](**kwargs)


@register_model_class
@register_analysis_network
@register_synthesis_network
@register_hybrid_cqnode
@register_compression_module
class IdentityLayer(nn.Identity):
    """

    """

    def __init__(self, **kwargs):
        super(IdentityLayer, self).__init__()
