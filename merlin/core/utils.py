from typing import Generator, Union
from torch import Tensor
from itertools import combinations, chain
import torch 
from functools import reduce

def generate_all_fock_states(m, n, no_bunching = False, loss = False) -> Generator:
    """
    Generates all possible Fock states for m modes and n photons.
    
    Args:
        m: Number of modes 
        n: Number of photons 
        no_bunching: If `True`, only maximum one photon in each mode.
        loss: Considers all lossy states.
    
    """
    if loss:
        # Combine all generate_all_fock_states from n to 0.
        yield from chain.from_iterable(
            generate_all_fock_states(m, n_, no_bunching=no_bunching, loss=False) 
            for n_ in reversed(range(n + 1)))
        return
    
    if no_bunching:
        if n > m or n < 0:
            return
        for positions in combinations(range(m), n):
            fock_state = [0] * m

            for pos in positions:
                fock_state[pos] = 1
            yield tuple(fock_state)

    else:
        if n == 0:
            yield (0,) * m
            return
        if m == 1:
            yield (n,)
            return

        for i in reversed(range(n + 1)):
            for state in generate_all_fock_states(m-1, n-i):
                yield (i,) + state


def prob_distribution_tensor_product(
    keys: Union[list[torch.Tensor]],
    *probs: torch.Tensor,
    merge_modes = False,
):
    """ 
    Takes the tensor product of two probability distributions. Based on
    `perceval.utils.statevector.BSDistribution.list_tensor_product`.
    
    Args:
        keys: Stack of states
        probs: Input probability distributions.
    Returns:
        Tuple of new keys and new corresponding probabilities. If keys 
        are given as Tensor, then a Tensor is returned instead.
    
    >>> keys1, probs1 = [(1, 0), (0, 1)], torch.tensor([0.5, 0.5])
    >>> 
    >>> keys2, probs2 = [(1, 0)], torch.tensor([1.0])
    >>> dist2 = (keys2, probs2)
    >>> print(prob_distribution_tensor_product([keys1, keys2], probs1, probs2, merge_modes=True))
    [(2, 0), (1, 1)], tensor([0.5000, 0.5000])
    """
    if len(probs[0].shape) == 1:
        probs = reduce(lambda acc, x: acc + (x.unsqueeze(0),), probs, ())

        batched_input = False
    else:
        batched_input = True
    
    num_probs = len(probs)
    num_batches = probs[0].size(0)
    
    if len(keys) != len(probs):
        raise ValueError(
            f"Invalid probability distribution for different length keys "
            f"({len(keys)}) & probs ({len(probs)})")
        
    if num_probs == 1:
        return keys[0], probs[0]
        
    if merge_modes:
        def _cartesian_sum(p1, p2):
            return (p1.unsqueeze(1) + p2.unsqueeze(0)).view(-1, p1.shape[1])
        
        new_keys = reduce(_cartesian_sum, keys)
    else:
        raise NotImplementedError()
    
    # Cartesian product of every pair of probs
    def _cartesian_product(p1, p2):
        output = (p1.unsqueeze(-1) * p2.unsqueeze(-2))
        return output.flatten(start_dim=-2) 
    
    # Unsqueeze each input tensor
    probs = reduce(lambda acc, x: acc + (x.unsqueeze(0),), probs, ())
    new_probs = reduce(_cartesian_product, probs).view(num_batches, -1)
    
    if merge_modes:
        # Remove duplicated keys & sum corresponding probs
        new_keys, inverse_idx = torch.unique(new_keys, dim=0, return_inverse=True)
        inverse_idx = inverse_idx.unsqueeze(0).expand(num_batches, -1)
        new_probs = torch.zeros(num_batches, len(new_keys)).scatter_add_(1, inverse_idx, new_probs)

    if not batched_input:
        new_probs = new_probs.squeeze(0)
    
    return new_keys.flip(0), new_probs.flip(1)
