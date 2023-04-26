import os
import re
from typing import Any, List, Mapping, Union

from pennylane import numpy as np
import pennylane as qml
import torch

from torch import nn
from torch.nn import functional as F
from torchdistill.common.constant import def_logger
from torchdistill.models.registry import register_model_class

from model.registry import get_hybrid_cqnode, get_q_circuit, register_hybrid_cqnode

logger = def_logger.getChild(__name__)

# Pennylane API makes it annoying to flexibly configure circuit parameters
qubits = int(os.environ.get("QUBITS", 4))
q_depth = int(os.environ.get("Q_DEPTH", 6))
q_delta = 0.01


@register_hybrid_cqnode
class HybridQNN(nn.Module):
    """
        Classical-Quantum Hybrid QNN model used for experiments in Paper
    """

    def __init__(self, in_features: int, circuit_name: str, use_residual: bool = False, draw_circuit: bool = True):
        super().__init__()
        self.circuit = get_q_circuit(circuit_name)
        self.qubits = qubits
        self.enc = nn.Linear(in_features, qubits)
        self.skip = nn.Linear(in_features, qubits) if use_residual else nn.Identity()
        self.use_residual = use_residual
        self.q_params = torch.nn.init.xavier_uniform_(nn.Parameter(torch.randn(qubits, q_depth)),
                                                      gain=1.0)

        if draw_circuit:
            logger.info(
                f"Drawing circuit {circuit_name}\n{qml.draw(self.circuit)(torch.randn(qubits), torch.randn(qubits * q_depth))}")

    def forward(self, x):
        skip = x
        x = self.enc(x)
        q_in = torch.tanh(x) * np.pi / 2.0
        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.empty(0, qubits, device=x.device)
        # pennylane pls proper batch support, even with 16 qubits it barely uses GPU resources to parallelize workload
        for elem in q_in:
            q_out_elem = self.circuit(elem, self.q_params).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        if self.use_residual:
            return self.skip(skip) + q_out
        return q_out


@register_hybrid_cqnode
class StackedHybridQNN(HybridQNN):
    """
        Could be interesting, was out of scope
    """

    def __init__(self,
                 in_features: int,
                 circuit_name: Union[str, List[str]],
                 use_residual: bool = False):
        if isinstance(circuit_name, str):
            circuit_name = [s.strip() for s in re.sub(r'[\[\]]', '', circuit_name).split(',')]
        self.no_stacks = len(circuit_name)
        circuit_name_dict = {i: circuit_name.count(i) for i in circuit_name}
        for name in circuit_name:
            logger.info(f"Stacking {name} {circuit_name_dict[name]} times..")
            logger.info(
                f"Drawing circuit {name}\n{qml.draw(get_q_circuit(name))(torch.randn(qubits), torch.randn(qubits * q_depth))}")
        first_circuit_name = circuit_name[0]
        super().__init__(in_features, first_circuit_name, use_residual, draw_circuit=False)
        # self.skips = nn.ModuleList([self.skip])
        q_params = self.q_params
        delattr(self, 'q_params')
        first_circuit = self.circuit
        delattr(self, 'circuit')
        self.circuits = [first_circuit]
        self.q_params = nn.ParameterList([q_params])
        for i in range(1, self.no_stacks):
            self.circuits.append(get_q_circuit(circuit_name[i]))
            # self.skips.append(nn.Linear(qubits, qubits) if use_residual else nn.Identity())
            self.q_params.append(torch.nn.init.xavier_uniform_(nn.Parameter(torch.randn(qubits, q_depth)),
                                                               gain=1.0))

    def forward(self, x):
        skip = x
        x = self.enc(x)
        q_in = torch.tanh(x) * np.pi / 2.0

        q_out = None
        for i in range(self.no_stacks):
            q_out = torch.empty(0, qubits, device=x.device)
            q_params = self.q_params[i]
            for elem in q_in:
                q_out_elem = self.circuits[i](elem, q_params).float().unsqueeze(0)
                q_out = torch.cat((q_out, q_out_elem))

            # if self.use_residual:
            #     q_in = self.skips[i](skip) + q_out
            #     q_in = torch.tanh(q_in) * np.pi / 2.0
            #
            # skip = q_out
        if self.use_residual:
            return self.skip(skip) + q_out
        return q_out


@register_hybrid_cqnode
class CirceptionHybridQNN(HybridQNN):
    """
    Could be interesting, was out of scope. TODO: Should initialize multiple devices
    """

    def __init__(self,
                 in_features: int,
                 circuit_name: Union[str, List[str]]):
        if isinstance(circuit_name, str):
            if ',' in circuit_name:
                circuit_name = [s.strip() for s in re.sub(r'[\[\]]', '', circuit_name).split(',')]
        self.no_stacks = len(circuit_name)
        circuit_name_dict = {i: circuit_name.count(i) for i in circuit_name}
        for name, occ in circuit_name_dict.items():
            logger.info(f"Stacking {name} {occ} times..")
            logger.info(
                f"Drawing circuit {name}\n{qml.draw(get_q_circuit(name))(torch.randn(qubits), torch.randn(qubits * q_depth))}")
        first_circuit_name = circuit_name[0]
        super().__init__(in_features, first_circuit_name, use_residual=True, draw_circuit=False)
        first_circuit = self.circuit
        delattr(self, 'circuit')
        self.circuits = [first_circuit]
        q_params = self.q_params
        delattr(self, 'q_params')
        self.q_params = nn.ParameterList([q_params])
        for i in range(1, self.no_stacks):
            self.circuits.append(get_q_circuit(circuit_name[i]))
            self.q_params.append(torch.nn.init.xavier_uniform_(nn.Parameter(torch.randn(qubits, q_depth)),
                                                               gain=1.0))
        self.pool = nn.Linear(qubits * self.no_stacks, qubits)

    def forward(self, x):
        skip = x
        x = self.enc(x)
        q_in = torch.tanh(x) * np.pi / 2.0

        embed_pool = []
        for i in range(self.no_stacks):
            q_out = torch.empty(0, qubits, device=x.device)
            q_params = self.q_params[i]
            q_in_circ = q_in
            for elem in q_in_circ:
                q_out_elem = self.circuits[i](elem, q_params).float().unsqueeze(0)
                q_out = torch.cat((q_out, q_out_elem))
            embed_pool.append(q_out)
        res = torch.cat(embed_pool, dim=1)
        res = F.leaky_relu(self.pool(res), negative_slope=0.1)
        return self.skip(skip) + res


@register_model_class
class HybridCQPredictor(nn.Module):
    """
        Embed Layer (Classic) --> Embed Layer (Quantum) --> Classifier (Quantum or Classic)
    """

    def __init__(self,
                 hybrid_cq_node_config: Mapping[str, Any],
                 no_classes: int):
        """

        :param q_device_config: params for physical quantum device or simulation
        :param classifier_config: params for classical and quantum layer
        :param classifier_config: params for classifier
        """
        super(HybridCQPredictor, self).__init__()
        self.embed: HybridQNN = get_hybrid_cqnode(node_name=hybrid_cq_node_config["name"],
                                                  **hybrid_cq_node_config["params"])
        self.classifier = nn.Linear(in_features=qubits, out_features=no_classes)

    def forward(self, x):
        features = self.embed(x)
        scores = self.classifier(features)
        return scores
