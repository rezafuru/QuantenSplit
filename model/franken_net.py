from collections import OrderedDict
from typing import Any, Container, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torchdistill.common.constant import def_logger
from torchdistill.models.registry import get_model, register_model_class, register_model_func

from model.compression.compression_module import CompressionModule
from model.cq_hybrid.hybrid_cq import HybridCQPredictor
from model.registry import get_compression_module

logger = def_logger.getChild(__name__)


@register_model_class
class CQHybridFrankenNet(nn.Module):
    def __init__(self,
                 compression_module: CompressionModule,
                 backbone: nn.Module,
                 predictors: Dict[str, HybridCQPredictor],
                 default_predictor: Optional[str] = None):
        super(CQHybridFrankenNet, self).__init__()

        self.compression_module: CompressionModule = compression_module
        self.backbone: nn.Module = backbone
        self.predictors: nn.ModuleDict[str, HybridCQPredictor] = nn.ModuleDict(predictors)
        # initialize classical counterparts to hybrid classifiers
        self._classical_predictors: nn.ModuleDict[str, nn.Module] = nn.ModuleDict({
            name: get_model("Mlp", **{"in_features": pred.embed.enc.in_features,
                                      "hidden_features": pred.embed.qubits,
                                      "out_features": pred.classifier.out_features})
            for name, pred in self.predictors.items()})
        self.compressor_updated: bool = False
        self._use_ensemble: bool = False
        self._active_predictor: Optional[Union[HybridCQPredictor, nn.Linear]] = default_predictor or nn.Identity()

    @property
    def use_ensemble(self) -> bool:
        return self._use_ensemble

    @use_ensemble.setter
    def use_ensemble(self, use_ensemble: bool):
        self._use_ensemble = use_ensemble
        if use_ensemble:
            logger.info("Using ensemble of predictors")
        else:
            logger.info("Using single predictor")

    @property
    def active_predictor(self) -> Union[HybridCQPredictor, nn.Linear]:
        return self._active_predictor

    @active_predictor.setter
    def active_predictor(self, predictor: str):
        assert predictor is None or predictor in self.predictors.keys(), f"{predictor} not registered"
        logger.info(f"Setting active predictor to: {predictor}")
        self._active_predictor = self.predictors[predictor]
        self._active_predictor.tag = predictor
        self._classical_predictors[predictor].tag = predictor

    def register_predictor(self, name: str, model: HybridCQPredictor):
        self.predictors[name] = model
        self._classical_predictors[name] = get_model("Mlp", **{"in_features": model.embed.enc.in_features,
                                                               "hidden_features": model.embed.qubits,
                                                               "oute_features": model.classifier.out_features})

    def toggle_classical(self):
        tag = self._active_predictor.tag
        if isinstance(self._active_predictor, HybridCQPredictor):
            self._active_predictor = self._classical_predictors[tag]
            logger.info(f"Activating classical predictor for {tag}")
        else:
            self._active_predictor = self.predictors[tag]

    def delete_backbone_predictor(self):
        logger.info("Replacing backbone classifier with identity")
        if hasattr(self.backbone, "head"):
            del self.backbone.head
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, "fc"):
            del self.backbone.fc
            self.backbone.fc = nn.Identity()

    def load_state_dict(self,
                        state_dict: Mapping[str, Any],
                        include_predictors: bool = False,
                        **kwargs):
        if include_predictors and "predictors" in state_dict:
            logger.info("Loading predictors")
            self.load_predictors_state_dict(state_dict.get("predictors"))
        if "compressor" in state_dict:
            logger.info("Loading compressor")
            self.load_compressor_state_dict(state_dict.get("compressor"))
        if "backbone" in state_dict:
            logger.info("Loading backbone")
            self.load_backbone_state_dict(state_dict.get("backbone"))

    def load_compressor_state_dict(self, state_dict: Any, **kwargs):
        self.compression_module.load_state_dict(state_dict)
        super().load_state_dict(state_dict, strict=False)

    def load_predictors_state_dict(self, state_dicts: Mapping[str, Any], **kwargs):
        assert set(state_dicts["hybrid"].keys()) == set(
            state_dicts["classical"].keys()), "classical hybrid mismatch"
        assert set(state_dicts["hybrid"].keys()).issubset(
            set(self.predictors.keys())), "Cannot map stored predictor weights"
        for name, predictor_weights in state_dicts["hybrid"].items():
            self.predictors[name].load_state_dict(predictor_weights)
            self._classical_predictors[name].load_state_dict(state_dicts["classical"].get(name))

    def load_backbone_state_dict(self, state_dict: Any, **kwargs):
        self.backbone.load_state_dict(state_dict)

    def update(self, force: bool = False) -> bool:
        if not isinstance(self.compression_module, CompressionModule):
            return False
        else:
            updated = self.compression_module.update(force=force)
            self.compressor_updated = True
            return updated

    def compress(self, obj: Tensor) -> Tuple[List[str], Container[int]]:
        return self.compression_module.compress(obj)

    def decompress(self, compressed_obj) -> Tensor:
        return self.compression_module.decompress(compressed_obj)

    def forward_compress(self) -> bool:
        return self.compressor_updated and not self.training

    def forward(self, x: Tensor) -> Tensor:
        if self.use_ensemble:
            return self.forward_ensemble(x)
        if self.forward_compress():
            compressed_obj = self.compression_module.compress(x)
            z = self.compression_module.decompress(compressed_obj)
        else:
            z = self.compression_module(x)
        features = self.backbone(z)
        scores = self.active_predictor(features)
        return scores

    def forward_classical(self, x: Tensor) -> Tensor:
        """

        """
        predictor = self._active_predictor.tag
        z = self.compression_module(x)
        features = self.backbone(z)
        scores = self._classical_predictors[predictor](features)
        return scores


    def to(self, device: str) -> nn.Module:
        super().to(device)
        for pred_h, pred_c in zip(self.predictors.values(), self._classical_predictors.values()):
            pred_h.to(device)
            pred_c.to(device)
        return self


@register_model_func
def franken_net(compression_module_config: Mapping[str, Any],
                backbone_module_config: Mapping[str, Any],
                predictors_configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
                circuit_name: Optional[str] = None,
                **kwargs) -> CQHybridFrankenNet:
    def _overwrite_circuit(new_circuit, config):
        config["hybrid_cq_node_config"]["params"]["circuit_name"] = new_circuit
        return config

    compression_module = get_compression_module(compression_module_config["name"],
                                                **compression_module_config["params"])
    backbone_module = get_model(backbone_module_config["name"], **backbone_module_config["params"])
    if predictors_configs:
        if circuit_name:
            logger.info(f"Loading circuit: {circuit_name} for all predictors")
        predictors = {alias: get_model(model_name=model_name,
                                       **(_overwrite_circuit(circuit_name,
                                                             config) if circuit_name else config)) for
                      alias, model_name, config in
                      predictors_configs[0]}
    else:
        logger.info("No predictors registered")
        predictors = dict()

    return CQHybridFrankenNet(compression_module=compression_module,
                              backbone=backbone_module,
                              predictors=predictors)
