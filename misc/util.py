import os
import uuid
from pathlib import Path
from logging import FileHandler, Formatter
from typing import Any, Dict, Mapping, Tuple, Union, Optional

import torch
from torch import Tensor, nn
from torchdistill.common import module_util
from torchdistill.common.constant import def_logger, LOGGING_FORMAT
from torchdistill.common.main_util import save_on_master
from torchdistill.models.official import get_image_classification_model
from torchdistill.models.registry import get_model

from torchinfo import summary
from model.franken_net import CQHybridFrankenNet

logger = def_logger.getChild(__name__)


def short_uid() -> str:
    return str(uuid.uuid4())[0:8]


def calc_compression_module_sizes(bnet_injected_model: nn.Module,
                                  device: str,
                                  input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
                                  log_model_summary: bool = True) -> Tuple[str, int, Dict[str, int]]:
    """
        Calc params and sizes individual components of compression module

        Returns (summary string, #params model, #params of the encoder)
    """
    assert hasattr(bnet_injected_model, 'compression_module'), "Model has no compression module"
    model_summary = summary(bnet_injected_model, input_size=input_size,
                            col_names=['input_size', 'output_size', 'mult_adds', 'num_params'],
                            depth=5,
                            device=device,
                            verbose=0,
                            mode="eval")
    model_params = model_summary.total_params
    if log_model_summary:
        logger.info(f"Bottleneck Injected model params:\n{model_summary}")

    # compression module core
    p_analysis = summary(bnet_injected_model.compression_module.g_a, col_names=["num_params"],
                         verbose=0,
                         mode="eval",
                         device=device).total_params
    p_synthesis = summary(bnet_injected_model.compression_module.g_s, col_names=["num_params"],
                          verbose=0,
                          mode="eval",
                          device=device).total_params

    # compression modules with side information
    p_hyper_analysis = summary(bnet_injected_model.compression_module.h_a,
                               col_names=["num_params"],
                               verbose=0,
                               mode="eval").total_params if hasattr(bnet_injected_model.compression_module,
                                                                    "h_a") else 0
    p_hyper_synthesis = summary(bnet_injected_model.compression_module.h_s,
                                col_names=["num_params"],
                                verbose=0,
                                mode="eval").total_params if hasattr(bnet_injected_model.compression_module,
                                                                     "h_s") else 0
    # compression modules with context models
    p_context_prediction = summary(bnet_injected_model.compression_module.context_prediction,
                                   col_names=["num_params"],
                                   verbose=0,
                                   mode="eval").total_params if hasattr(bnet_injected_model.compression_module,
                                                                        "context_prediction") else 0
    p_entropy_parameters = summary(bnet_injected_model.compression_module.entropy_parameters,
                                   col_names=["num_params"],
                                   verbose=0,
                                   mode="eval").total_params if hasattr(bnet_injected_model.compression_module,
                                                                        "entropy_parameters") else 0

    # entropy estimation
    params_eb = summary(bnet_injected_model.compression_module.entropy_bottleneck, col_names=["num_params"],
                        verbose=0,
                        mode="eval").total_params
    params_comp_module = summary(bnet_injected_model.compression_module, col_names=["num_params"],
                                 verbose=0).total_params
    # params_comp_module += p_reconstruction
    summary_str = f"""
                Compression Module Summary: 
                Params Analysis: {p_analysis:,}
                Params Synthesis: {p_synthesis:,}
                Params Hyper Analysis: {p_hyper_analysis:,}
                Params Hyper Synthesis: {p_hyper_synthesis:,}
                Params Context Prediction: {p_context_prediction:,}
                Params Entropy Parameters: {p_entropy_parameters :,}   
                Params Entropy Bottleneck: {params_eb:,}
                Total Params Compression Module: {params_comp_module:,}

                Which makes up {params_comp_module / model_params * 100:.2f}% of the total model params

                """

    enc_params_main = p_analysis
    enc_params_hyper = p_hyper_analysis + p_hyper_synthesis
    enc_params_context_module = p_entropy_parameters + p_context_prediction
    total_encoder = enc_params_main + enc_params_hyper + enc_params_context_module
    return summary_str, model_params, {"Main Network": enc_params_main,
                                       "Hyper Network": enc_params_hyper,
                                       "Context Module": enc_params_context_module,
                                       "Total Encoder Params": total_encoder}


def save_ckpt_fs(model: CQHybridFrankenNet,
                 output_file_path: str,
                 store_backbone: bool = False):
    make_parent_dirs(output_file_path)
    state_dict = {'model': {'compressor': model.compression_module.state_dict(),
                            'predictors':
                                {"hybrid": {name: predictor.state_dict() for name, predictor in
                                            model.predictors.items()},
                                 "classical": {name: predictor.state_dict() for name, predictor in
                                               model._classical_predictors.items()}}}
                  }
    if store_backbone:
        state_dict['backbone'] = model.backbone.state_dict()
    save_on_master(state_dict, output_file_path)


def load_fs_weights(ckpt_file_path: Optional[Union[str, bytes, os.PathLike]],
                    model: CQHybridFrankenNet,
                    strict: bool = True,
                    predictors_only: bool = False) -> Union[nn.Module, CQHybridFrankenNet]:
    if os.path.exists(ckpt_file_path):
        ckpt = torch.load(ckpt_file_path, map_location="cpu")
        if predictors_only:
            logger.info("Loading only predictors parameters")
            model.load_predictors_state_dict(ckpt["model"]["predictors"], strict=strict)
        else:
            logger.info("Loading model parameters")
            model.load_state_dict(ckpt["model"], strict=strict)
    else:
        logger.warning("ckpt path not found, not loading any weights")
    return model


def load_model(model_config: Mapping[str, Any],
               device: str,
               distributed: bool = False,
               skip_ckpt: bool = False) -> Union[nn.Module, CQHybridFrankenNet]:
    model = get_image_classification_model(model_config, distributed)
    if model is None:
        repo_or_dir = model_config.get("repo_or_dir", None)
        model = get_model(model_config["name"], repo_or_dir, **model_config['params'])
    if not skip_ckpt:
        ckpt_file_path = os.path.expanduser(model_config.get('ckpt'))
        load_fs_weights(ckpt_file_path, model=model, strict=True, predictors_only=False)
    else:
        logger.info('Skipping loading from checkpoint...')
    return model.to(device)


def compute_bitrate(likelihoods, input_size) -> Tuple[int, int]:
    b, _, h, w = input_size

    likelihoods = likelihoods.detach().cpu()
    bitrate = -likelihoods.log2().sum()
    bbp = bitrate / (b * h * w)
    return bbp, bitrate


def check_if_module_exits(module, module_path) -> bool:
    module_names = module_path.split('.')
    child_module_name = module_names[0]
    if len(module_names) == 1:
        return hasattr(module, child_module_name)

    if not hasattr(module, child_module_name):
        return False
    return check_if_module_exits(getattr(module, child_module_name), '.'.join(module_names[1:]))


def extract_entropy_bottleneck_module(model) -> nn.Module:
    model_wo_ddp = model.module if module_util.check_if_wrapped(model) else model
    entropy_bottleneck_module = None
    if check_if_module_exits(model_wo_ddp, "compression_module.entropy_bottleneck"):
        entropy_bottleneck_module = module_util.get_module(model_wo_ddp, "compression_module")
    elif check_if_module_exits(model_wo_ddp, 'compression_model.entropy_bottleneck'):
        entropy_bottleneck_module = module_util.get_module(model_wo_ddp, "compression_model")
    return entropy_bottleneck_module


def make_dirs(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def make_parent_dirs(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def setup_log_file(log_file_path, mode='w'):
    make_parent_dirs(log_file_path)
    fh = FileHandler(filename=log_file_path, mode=mode)
    fh.setFormatter(Formatter(LOGGING_FORMAT))
    def_logger.addHandler(fh)


def uniquify(path) -> str:
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def prepare_log_file(test_only, log_file_path, config_path, start_epoch, overwrite=False):
    eval_file = "_eval" if test_only else ""
    if log_file_path:
        log_file_path = f"{os.path.join(log_file_path, Path(config_path).stem)}{eval_file}.log"
    else:
        log_file_path = f"{config_path.replace('config', 'logs', 1)}{eval_file}".replace('.yaml', '.log', 1)
    qubits = os.environ.get("QUBITS")
    if qubits:
        log_file_path = Path(log_file_path).parent / f"qubits_{qubits}" / Path(log_file_path).name
    if start_epoch == 0 or overwrite:
        log_file_path = uniquify(log_file_path)
        mode = 'w'
    else:
        mode = 'a'
    setup_log_file(os.path.expanduser(log_file_path), mode=mode)


class Tokenizer(nn.Module):
    """
        Patch embed without Projection (From Image Tensor to Token Tensor)
    """

    def __init__(self):
        super(Tokenizer, self).__init__()

    def forward(self, x) -> Tensor:
        x = x.flatten(2).transpose(1, 2)  # B h*w C
        return x


class Detokenizer(nn.Module):
    """
        Inverse operation of Tokenizer (From Token Tensor to Image Tensor)
    """

    def __init__(self, spatial_dims):
        super(Detokenizer, self).__init__()
        self.spatial_dims = spatial_dims

    def forward(self, x) -> Tensor:
        B, _, C = x.shape
        H, W = self.spatial_dims
        return x.transpose(1, 2).view(B, -1, H, W)


def overwrite_config(org_config, sub_config):
    for sub_key, sub_value in sub_config.items():
        if sub_key in org_config:
            if isinstance(sub_value, dict):
                overwrite_config(org_config[sub_key], sub_value)
            else:
                org_config[sub_key] = sub_value
        else:
            org_config[sub_key] = sub_value
