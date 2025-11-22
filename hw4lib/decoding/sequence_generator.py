import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        x = x.to(self.device)
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        eos_id = self.tokenizer.eos_id

        # Generate step by step until max_length
        for _ in range(self.max_length - x.size(1)):
            # If all finished, stop early
            if finished.all():
                break

            # Get logits for next token
            logits = self.score_fn(x)  # (batch_size, vocab_size)

            # Apply repetition penalty if needed
            logits = self._apply_repeat_penalty(logits, x, penalty=repeat_penalty)

            # Temperature scaling
            logits = logits / temperature

            # Convert to log-probs
            log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # Greedy: take argmax
            next_tokens = torch.argmax(log_probs, dim=-1)  # (batch_size,)

            # Log-prob of chosen tokens
            token_log_probs = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            # Update scores only for sequences that are not yet finished
            scores = torch.where(finished, scores, scores + token_log_probs)

            # For already finished sequences, just keep appending EOS to keep shapes consistent
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, eos_id),
                next_tokens
            )

            # Append next token
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            # Update finished mask
            finished = finished | (next_tokens == eos_id)

        return x, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length) where each sequence in a beam set is sorted by score
             - scores is of shape (batch_size, beam_width)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        x = x.to(self.device)
        batch_size, start_len = x.size()
        vocab_size = None
        eos_id = self.tokenizer.eos_id

        # Initialize beams: repeat input across beams
        sequences = x.unsqueeze(1).repeat(1, beam_width, 1)  # (B, W, T)
        scores = torch.zeros(batch_size, beam_width, device=x.device)  # log-prob scores
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - start_len):
            # If all beams in all batches are finished, stop early
            if finished.all():
                break

            # Flatten beams for scoring: (B*W, T)
            flat_seqs = sequences.reshape(batch_size * beam_width, -1)

            # Get logits and reshape back to (B, W, V)
            logits = self.score_fn(flat_seqs)  # (B*W, V)
            if vocab_size is None:
                vocab_size = logits.size(-1)
            logits = logits.view(batch_size, beam_width, vocab_size)

            # Apply repetition penalty
            logits = self._apply_repeat_penalty(logits, sequences, penalty=repeat_penalty)

            # Temperature scaling
            logits = logits / temperature

            # Log-probs for all candidate next tokens
            log_probs = torch.log_softmax(logits, dim=-1)  # (B, W, V)

            # For finished beams: force them to keep emitting EOS with prob=1 (log_prob=0)
            if finished.any():
                B, W, V = log_probs.size()
                log_probs_flat = log_probs.view(B * W, V)
                finished_flat = finished.view(B * W)

                log_probs_flat[finished_flat] = float('-inf')
                log_probs_flat[finished_flat, eos_id] = 0.0

                log_probs = log_probs_flat.view(B, W, V)

            # Candidate scores for every (beam, token)
            candidate_scores = scores.unsqueeze(-1) + log_probs  # (B, W, V)

            # Flatten beams and tokens, pick top beam_width per batch
            candidate_scores_flat = candidate_scores.view(batch_size, -1)  # (B, W*V)
            topk_scores, topk_indices = torch.topk(candidate_scores_flat, beam_width, dim=-1)  # (B, W)

            # Map flat indices back to (prev_beam_idx, token_idx)
            vocab_size = log_probs.size(-1)
            prev_beam_indices = topk_indices // vocab_size  # (B, W)
            token_indices = topk_indices % vocab_size       # (B, W)

            # Gather previous sequences according to chosen beams
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(-1).expand(-1, beam_width)
            chosen_prev_seqs = sequences[batch_indices, prev_beam_indices]  # (B, W, cur_len)

            # Append chosen tokens
            new_sequences = torch.cat(
                [chosen_prev_seqs, token_indices.unsqueeze(-1)],
                dim=-1
            )  # (B, W, cur_len+1)

            # Update finished mask based on previous finish + EOS just generated
            new_finished = finished[batch_indices, prev_beam_indices] | (token_indices == eos_id)

            # Commit updates
            sequences = new_sequences
            scores = topk_scores
            finished = new_finished

        # sequences: (B, W, seq_len), scores: (B, W)
        return sequences, scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break

            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            # Check if any sequence has reached EOS 
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]
