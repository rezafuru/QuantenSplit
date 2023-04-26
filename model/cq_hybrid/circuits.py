import os

import pennylane as qml

from model.cq_hybrid.gates import H_layer, RX_layer, RY_layer, RZ_layer, entangling_layer, entangling_layer_linear
from model.registry import register_q_circuit

# Pennylane API makes it annoying to flexibly configure circuit parameters
qubits = int(os.environ.get("QUBITS", 4))
q_depth = int(os.environ.get("Q_DEPTH", 6))
q_device = os.environ.get("Q_DEVICE", "default.qubit")
q_delta = 0.01

dev = qml.device(q_device, wires=qubits)


@register_q_circuit
@qml.qnode(dev, interface="torch")
def mari_quantum_circuit(q_input_features, q_weights_flat):
    """
        The variational quantum circuit from Mari et al.
        https://arxiv.org/abs/1912.08278
    """

    q_weights = q_weights_flat.reshape(q_depth, qubits)

    H_layer(qubits)

    RY_layer(q_input_features)

    for k in range(q_depth):
        entangling_layer(qubits)
        RY_layer(q_weights[k])

    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(qubits)]
    return tuple(exp_vals)


@qml.qnode(dev, interface="torch")
def alternating_rotation_circuit(q_input_features, q_weights_flat):
    """
        The variational quantum circuit used in the Paper
    """

    q_weights = q_weights_flat.reshape(q_depth, qubits)

    H_layer(qubits)

    RZ_layer(q_input_features)

    # Sequence of parameterized layers with alternating rotations
    for k in range(q_depth):
        entangling_layer(qubits)
        if k % 2 == 0:
            RY_layer(q_weights[k])
        else:
            RZ_layer(q_weights[k])

    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(qubits)]
    return tuple(exp_vals)


@register_q_circuit
@qml.qnode(dev, interface="torch")
def alternating_rotation_circuit_a1(q_input_features, q_weights_flat):

    q_weights = q_weights_flat.reshape(q_depth, qubits)

    RY_layer(q_input_features)

    for k in range(q_depth):
        entangling_layer_linear(qubits)
        if k % 2 == 0:
            RZ_layer(q_weights[k])
        else:
            RY_layer(q_weights[k])

    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(qubits)]
    return tuple(exp_vals)



@register_q_circuit
@qml.qnode(dev, interface="torch")
def alternating_rotation_circuit_a2(q_input_features, q_weights_flat):
    q_weights = q_weights_flat.reshape(q_depth, qubits)


    RY_layer(q_input_features)

    for k in range(q_depth):
        entangling_layer(qubits)
        if k % 2 == 0:
            RZ_layer(q_weights[k])
        else:
            RY_layer(q_weights[k])

    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(qubits)]
    return tuple(exp_vals)
