import torch 
import numpy as np 
import pytest
import perceval as pcvl

from ..core.quantum_kernels import FeatureMap, FidelityKernel

@pytest.mark.parametrize("x1, x2", [
    (torch.tensor([0.5]), torch.tensor([0.5])),
    (torch.tensor(0.5), torch.tensor(0.5)),
    (0.5, 0.5),
    (np.array([0.5]), np.array([0.5])),
    (np.array(0.5), np.array(0.5)),
    (torch.rand(2, 1), torch.rand(2, 1)),
    (torch.rand(1, 1), torch.rand(1, 1)),
]
)
def test_input_size_one(x1, x2):
    """Test a quantum kernel with a single input parameter"""
    input_state = [1, 0]
    x = pcvl.P("x1")
    
    # Test non-trainable feature map
    circuit = pcvl.Circuit(2) // pcvl.PS(x) // pcvl.BS()
    feature_map = FeatureMap(circuit, input_size=1, input_parameters="x")
    quantum_kernel = FidelityKernel(feature_map, input_state)
    quantum_kernel(x1, x2)
    
    # Test trainable feature map
    A1, A2 = pcvl.P("A1"), pcvl.P("A2")
    circuit = pcvl.Circuit(2) // pcvl.BS(A1) // pcvl.PS(x) // pcvl.BS(A2)

    feature_map = FeatureMap(
        circuit, 
        input_size=1, 
        input_parameters="x",
        trainable_parameters=["A"]
    )
    quantum_kernel = FidelityKernel(feature_map, input_state)
    quantum_kernel(x1, x2)


@pytest.mark.parametrize("x1, x2", [
    (torch.rand(4, 2), torch.rand(4, 2)),
    (torch.rand(1, 2), torch.rand(1, 2)),
    (torch.rand(2), torch.rand(2)),
    (np.random.rand(4, 2), np.random.rand(4, 2)),
    (np.random.rand(1, 2), np.random.rand(1, 2)),
    (np.random.rand(2), np.random.rand(2)),
])
def test_multiple_inputs(x1, x2):
    """Check that fidelity quantum kernel works with two data embedding params"""
    input_state = [1, 0]
    X1, X2 = pcvl.P("x1"), pcvl.P("x2")
    
    # Test non-trainable feature map
    circuit = pcvl.Circuit(2) // pcvl.PS(X1) // pcvl.BS() // pcvl.PS(X2) // pcvl.BS()
    feature_map = FeatureMap(circuit, input_size=2, input_parameters="x")
    quantum_kernel = FidelityKernel(feature_map, input_state)
    quantum_kernel(x1, x2)
    
    # Test trainable feature map
    A1, A2 = pcvl.P("A1"), pcvl.P("A2")
    circuit = pcvl.Circuit(2) // pcvl.PS(X1) // pcvl.BS(A1) // pcvl.PS(X2) // pcvl.BS(A2)

    feature_map = FeatureMap(
        circuit, 
        input_size=2, 
        input_parameters="x",
        trainable_parameters=["A"]
    )    
    quantum_kernel = FidelityKernel(feature_map, input_state)
    quantum_kernel(x1, x2)

