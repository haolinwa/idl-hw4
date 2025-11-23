import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Create a mask to identify non-padding positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
    Returns:
        A boolean mask tensor with shape (N, T), where: 
            - padding positions are marked with True 
            - non-padding positions are marked with False.
    """
    batchSize, seqLen = padded_input.shape[:2]

    lengths = input_lengths.to(padded_input.device)

    # Support both 1D length tensors and higher-dimensional inputs (e.g.,
    # per-timestep indicators). In the latter case, reduce across all
    # non-batch dimensions to recover one length per sequence.
    if lengths.dim() > 1:
        if lengths.shape[0] != batchSize:
            raise ValueError(
                f"input_lengths has incompatible batch dimension: expected {batchSize}, got {lengths.shape[0]}"
            )
        reduce_dims = list(range(1, lengths.dim()))
        lengths = lengths.sum(dim=reduce_dims)

    # If a flat tensor is provided but doesn't match the batch dimension,
    # attempt to reshape using the known batch size. This handles cases where
    # lengths arrive as a flattened per-timestep indicator (batch_size * seq_len).
    if lengths.dim() == 1 and lengths.numel() != batchSize:
        if lengths.numel() % batchSize != 0:
            raise ValueError(
                f"input_lengths size {lengths.numel()} is incompatible with batch size {batchSize}"
            )
        lengths = lengths.view(batchSize, -1).sum(dim=1)

    lengths = lengths.view(batchSize, 1)

    idx = torch.arange(seqLen, device=padded_input.device).view(1, seqLen)
    mask = idx >= lengths
    
    return mask

''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Create a mask to identify non-causal positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
    
    Returns:
        A boolean mask tensor with shape (T, T), where: 
            - non-causal positions (don't attend to) are marked with True 
            - causal positions (can attend to) are marked with False.
    """
    # TODO: Implement CausalMask
    T = padded_input.shape[1]
    device = padded_input.device

    # upper-triangular (excluding diagonal) are non-causal (True)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    return mask

