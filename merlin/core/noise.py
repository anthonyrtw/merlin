import math
from typing import Tuple
from torch import Tensor
import torch

from perceval import NoiseModel, Source

from .utils import prob_distribution_tensor_product, generate_all_fock_states
from ..pcvl_pytorch import build_slos_distribution_computegraph as build_slos_graph

class PartialDistinguishabilitySLOS:
    """ 
    Equivalent to SLOSGraph but with partial distinguishability.
    
    
    """
    def __init__(self, input_state: list, indistinguishability):
        self.input_state = input_state 
        self.indistinguishability = indistinguishability
        
        if max(input_state) > 1:
            raise NotImplementedError(
                'States with multiple photons per mode not supported yet.')
        
        #self._partial_input_states = self._generate_all_partial_states()
        
        m, self.n = len(input_state), sum(input_state)
        self._slos_graphs = [build_slos_graph(m, n_i) 
                            for n_i in range(1, self.n + 1)]
        
        # Amplitudes of good & bad bits respectively
        self.g = math.sqrt(self.indistinguishability)
        self.b = 1 - math.sqrt(self.indistinguishability)
        
        self._keys_per_n = [
            torch.tensor(list(generate_all_fock_states(m, i))) 
            for i in range(1, self.n + 1)
        ]
        
        self._decomposed_input = self._construct_decomposed_input()
        
        self._state_info = []
        for (_, states) in self._decomposed_input:
            n_tensor = states.sum(dim=1)
            fock_states = [self._keys_per_n[n - 1] for n in n_tensor]
            
            self._state_info.append((n_tensor, fock_states))
            
    
    def compute(self, unitary):
        if unitary.size(0) == unitary.size(1) and unitary.ndim == 2:
            unitary = unitary.unsqueeze(0)
        b = len(unitary)

        partial_slos_results = {}
        output = torch.zeros(b, len(self._keys_per_n[-1]))
        
        for i, (amplitude, states) in enumerate(self._decomposed_input):
            probs_per_state = []
            n_tensor, fock_states = self._state_info[i]
            
            for j, state in enumerate(states):
                state_key = tuple(state.tolist())

                # Calculate slos results for each partial input
                if state_key not in partial_slos_results:
                    _, probs = self._slos_graphs[n_tensor[j] - 1].compute(unitary, state)
                    partial_slos_results[state_key] = probs
                else:
                    probs = partial_slos_results[state_key]
                    
                probs_per_state.append(probs)
            
            _, tensor_product_state = prob_distribution_tensor_product(
                fock_states, *probs_per_state, merge_modes=True
            )
            output += amplitude * tensor_product_state
        
        return self._keys_per_n[-1], output


    def _construct_decomposed_input(self):
        """ 
        The construction of the decomposed input is incredibly messy. Apologies.
        
        Given an input state, all possible collection of partial states & 
        their amplitude corresponding to the indistinguishability is 
        calculated.
        """
        bad_bits = self._generate_bad_bits()
        nonzero = torch.where(bad_bits != 0, bad_bits, torch.ones_like(bad_bits))  
        amplitudes = nonzero.prod(dim=1) * (bad_bits != 0).any(dim=1)

        decomposed_input = []
        for i, state in enumerate(bad_bits):
            state = self._convert_bad_bits(state) 
            
            state = torch.where((state != 0) & (state != 1), torch.ones_like(state), state)
            state = state.to(torch.int)
            
            weights = 2 ** torch.arange(state.size(1) - 1, -1, -1)
            values = state @ weights.to(torch.int32)
            sorted_indices = torch.argsort(values, descending=True)
            state = state[sorted_indices]
            
            decomposition = (amplitudes[i], state)
            decomposed_input.append(decomposition)
            
        decomposed_input = list(reversed(decomposed_input))
        return decomposed_input
    
    
    def _convert_bad_bits(self, bad_bits):
        n = len(bad_bits)
        bad_mask = (bad_bits != self.g) & (bad_bits != 0)
        bad_positions = torch.where(bad_mask)[0]
        n_bad = len(bad_positions)
        
        if n_bad == 0:
            return bad_bits.unsqueeze(0)
        
        # Create result tensor
        result = torch.zeros(n_bad + 1, n, dtype=bad_bits.dtype, device=bad_bits.device)
        
        # Use advanced indexing to set bad bits efficiently
        # Create identity-like matrix for bad positions
        row_indices = torch.arange(n_bad, device=bad_bits.device)
        result[row_indices, bad_positions] = -1
        
        # Set base state (preserve good bits, zero out bad bits)
        result[-1] = torch.where(bad_mask, 0, bad_bits)
        
        result = result[result.norm(dim=1) != 0]
        return result

    def _generate_bad_bits(self):
        """
        
        """
        input_tensor = torch.tensor(self.input_state, dtype=torch.float32)
        good_mask = (input_tensor == 1)
        good_indices = torch.where(good_mask)[0]
        n_good = len(good_indices)
        
        if n_good == 0:
            return input_tensor.unsqueeze(0)
        
        # Create all combinations using meshgrid
        choices = [torch.tensor([self.g, self.b], dtype=torch.float32) for _ in range(n_good)]
        grids = torch.meshgrid(*choices, indexing='ij')
        
        # Stack and reshape to get all combinations
        combinations = torch.stack([g.flatten() for g in grids], dim=1)
        n_combinations = combinations.shape[0]
        
        # Create result tensor
        result = input_tensor.unsqueeze(0).expand(n_combinations, -1).clone()
        result[:, good_indices] = combinations
        
        return result
    
    # def _generate_all_partial_states(self):
    #     """Generate all partial states for a given input state."""
    #     input_state = self.input_state
    #     if not isinstance(input_state, torch.Tensor):
    #         input_state = torch.tensor(input_state)
            
    #     idx = torch.nonzero(input_state).flatten()
    #     n = idx.numel()
    #     combinations = torch.stack(
    #         torch.meshgrid(*[torch.tensor([0, 1])] * n, indexing='ij'), dim=-1)
    #     combinations = combinations.reshape(-1, n).flip(0).to(torch.int)
        
    #     result = torch.zeros((combinations.size(0), input_state.size(0)), 
    #                         dtype=torch.int)
    #     result[:, idx] = combinations
        
    #     return result[:-1]


