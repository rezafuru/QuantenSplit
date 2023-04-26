import os
import pennylane as qml

# Pennylane API makes it annoying to flexibly configure circuit parameters


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RX_layer(w):
    """Layer of parametrized qubit rotations around the x axis.
    """
    for idx, element in enumerate(w):
        qml.RX(element, wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def RZ_layer(w):
    """Layer of parametrized qubit rotations around the z axis.
    """
    for idx, element in enumerate(w):
        qml.RZ(element, wires=idx)


def entangling_layer(nqubits):
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


def entangling_layer_linear(nqubits):
    for i in range(0, nqubits - 1):
        qml.CNOT(wires=[i, i + 1])


def apply_hadamard(qubits):
    for i in range(qubits):
        qml.Hadamard(wires=i)


def apply_parameterized_gate(gate, w):
    """
        Apply single qubit parameterized circuit
    """
    for i, e in enumerate(w):
        gate(e, wires=i)
