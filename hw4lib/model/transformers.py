import torch.nn as nn
import torch
import random
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding
from .decoder_layers import SelfAttentionDecoderLayer, CrossAttentionDecoderLayer
from .encoder_layers import SelfAttentionEncoderLayer
from .speech_embedding import SpeechEmbedding
import torch.nn.functional as F
import warnings
from torchinfo import summary

"""
TODO: Implement these Modules.

This file contains two key transformer architectures:

1. DecoderOnlyTransformer: Used for language modeling tasks (like GPT)
   - Contains a stack of SelfAttentionDecoderLayers
   - Uses causal masking to prevent attending to future tokens
   - Includes optional weight tying and layer dropout features

    Key components to implement:
    1. Token Embedding Layer: Convert token IDs to vectors
    2. Positional Encoding: Add position information
    3. Decoder Stack: Process tokens sequentially
    4. Output Projection: Convert final representations to logits

    Architecture follows Pre-LN (Layer Normalization) design where:
    - Layer normalization is applied at the start of each sublayer
    - Residual connections wrap around each sublayer
    - Final layer norm is applied before output projection

    Implementation Notes:
    1. The forward pass should handle:
    - Proper masking (both padding and causal)
    - Collecting attention weights from all layers
    - Optional layer dropout during training
    
    2. The score method should:
    - Handle single token prediction
    - Not apply padding masks
    - Return only the final token's logits

2. EncoderDecoderTransformer: Used for ASR (Automatic Speech Recognition) tasks
   - Contains an encoder stack for processing speech features
   - Contains a decoder stack for generating text tokens
   - Uses both self-attention and cross-attention mechanisms
   - Includes CTC auxiliary loss support and optional weight tying

   Key components to implement:
   1. Speech Embedding: Convert speech features to vectors with time reduction
   2. Positional Encoding: Add position information (optional for both encoder/decoder)
   3. Encoder Stack: Process speech features
   4. Decoder Stack: Generate text tokens
   5. CTC Head: For auxiliary CTC loss computation
   6. Output Projection: Convert final representations to logits

   Architecture follows Pre-LN (Layer Normalization) design where:
   - Layer normalization is applied at the start of each sublayer
   - Residual connections wrap around each sublayer
   - Final layer norm is applied before output projection

   Implementation Notes:
   1. The forward pass should handle:
   - Proper masking (padding for encoder, both padding and causal for decoder)
   - Collecting attention weights from all layers
   - Optional layer dropout during training
   - CTC logits computation

   2. The score method should:
   - Handle single token prediction given encoder output
   - Not apply padding masks to decoder inputs
   - Return only the final token's logits
"""


## -------------------------------------------------------------------------------------------------
## Decoder-Only Transformer
## -------------------------------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    """
    A Pre-LN Decoder-Only Transformer model.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        num_classes: int,
        weight_tying: bool = False,
        layer_drop_rate: float = 0.0,
    ):
        """
        Initialize the Decoder-Only Transformer model.

        Args:
            num_layers: int, number of decoder layers
            d_model: int, model dimension
            num_heads: int, number of attention heads
            d_ff: int, feed-forward dimension
            dropout: float, dropout rate
            max_len: int, maximum sequence length this model can handle
            num_classes: int, number of classes
            weight_tying: bool, whether to use weight tying (default: False)
            layer_drop_rate: float, layer drop rate (default: 0.0)
        """
        super().__init__()

        # Initialize the decoder
        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Create a ModuleList of decoder layers based on the number of layers
        self.dec_layers = nn.ModuleList(
            [
                SelfAttentionDecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Create target embedding and other layers
        self.target_embedding = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.final_linear = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Weight tying (extra form of regularization, read more about it)
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def forward(
        self,
        padded_targets: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass for the decoder. Used for Training only. Tokens are assumed to be right-padded.
        Args:
            padded_targets (torch.Tensor): The padded target sequence. shape: (batch_size, seq_len)
            target_lengths (Optional[torch.Tensor]): The lengths of the target sequences. shape: (batch_size,)
        Returns:
            seq_out (torch.Tensor): The output sequence. shape: (batch_size, seq_len, d_model)
            runnint_att (dict): The attention weights. shape: (batch_size, seq_len, seq_len)
        """
        # DO NOT MODIFY
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")

        # Create padding mask for padded_targets on the same device as the input (use PadMask)
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_targets, target_lengths)

        # Create causal mask to prevent attending to future tokens on the same device as the input (use CausalMask)
        causal_mask = CausalMask(padded_targets)

        # Apply the embedding
        x = self.target_embedding(padded_targets)
        # Apply positional encoding
        x = self.positional_encoding(x)
        # Apply dropout
        x = self.dropout(x)

        # Pass through all decoder layers, save attention masks
        runnint_att = {}
        for i in range(self.num_layers):
            # Optionally apply LayerDrop during training (More regularization!)
            if (
                self.training
                and self.layer_drop_rate > 0
                and random.random() < self.layer_drop_rate
            ):
                continue

            # Pass through decoder layer
            x, attention = self.dec_layers[i](
                x, key_padding_mask=pad_mask_dec, attn_mask=causal_mask
            )

            # Save attention weights
            runnint_att['layer{}_dec_self'.format(i + 1)] = attention

        # Apply normalization
        x = self.norm(x)
        # TODO: Linear layer (Final Projection) for next character prediction
        seq_out = self.final_linear(x)

        return seq_out, runnint_att

    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        """
        Score the tokens for the decoder.
        This is used for scoring the next token for a given prompt.
        Padding mask is not applied so ensure that the prompts are not padded.
        Can only handle batch_size = 1 or batch with same lengths and no padding.
        Args:
            prompts (torch.Tensor) : tensor of fixed length token sequences. shape: (batch_size, seq_len)
        Returns:
            logits (torch.Tensor): Batch of next token logits. shape: (batch_size, num_classes)
        """
        if self.training:
            raise ValueError(
                "score method is not supported during training, use forward method instead"
            )
        # Forward pass with no target lengths
        seq_out, _ = self.forward(batch_prompts, target_lengths=None)
        # Return the last token's logits for next token prediction
        logits = seq_out[:, -1, :]
        return logits


## -------------------------------------------------------------------------------------------------
## Encoder-Decoder Transformer
## -------------------------------------------------------------------------------------------------
class EncoderDecoderTransformer(nn.Module):
    """Pre-LN Encoder-Decoder Transformer with optional CTC head."""

    def __init__(
        self,
        input_dim: int,
        time_reduction: int,
        reduction_method: str,
        num_encoder_layers: int,
        num_encoder_heads: int,
        d_ff_encoder: int,
        num_decoder_layers: int,
        num_decoder_heads: int,
        d_ff_decoder: int,
        d_model: int,
        dropout: float,
        max_len: int,
        num_classes: int,
        weight_tying: bool = False,
        layer_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_classes = num_classes
        self.layer_drop_rate = layer_drop_rate

        # Embeddings
        self.source_embedding = SpeechEmbedding(
            input_dim=input_dim,
            output_dim=d_model,
            dropout=dropout,
            time_reduction=time_reduction,
            reduction_method=reduction_method,
        )
        self.target_embedding = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        # Encoder/Decoder stacks
        self.enc_layers = nn.ModuleList(
            [
                SelfAttentionEncoderLayer(
                    d_model=d_model,
                    num_heads=num_encoder_heads,
                    d_ff=d_ff_encoder,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dec_layers = nn.ModuleList(
            [
                CrossAttentionDecoderLayer(
                    d_model=d_model,
                    num_heads=num_decoder_heads,
                    d_ff=d_ff_decoder,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        self.final_linear = nn.Linear(d_model, num_classes)
        self.ctc_head = nn.Linear(d_model, num_classes)

        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def encode(self, source: torch.Tensor, source_lengths: torch.Tensor):
        enc_out, enc_lengths = self.source_embedding(source, source_lengths)
        enc_out = self.positional_encoding(enc_out)
        enc_out = self.dropout(enc_out)

        attn_weights = {}
        pad_mask_src = PadMask(enc_out, enc_lengths)

        for i, layer in enumerate(self.enc_layers):
            if (
                self.training
                and self.layer_drop_rate > 0
                and random.random() < self.layer_drop_rate
            ):
                continue
            enc_out, enc_attn = layer(enc_out, key_padding_mask=pad_mask_src)
            attn_weights[f"layer{i+1}_enc_self"] = enc_attn

        enc_out = self.encoder_norm(enc_out)

        ctc_logits = self.ctc_head(enc_out)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        ctc_input = {"log_probs": ctc_log_probs, "lengths": enc_lengths}

        return enc_out, pad_mask_src, attn_weights, ctc_input

    def decode(
        self,
        targets: torch.Tensor,
        encoder_output: torch.Tensor,
        target_lengths: Optional[torch.Tensor],
        pad_mask_src: torch.Tensor,
    ):
        x = self.target_embedding(targets)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        attn_weights = {}
        pad_mask_dec = PadMask(x, target_lengths) if target_lengths is not None else None
        causal_mask = CausalMask(targets)

        for i, layer in enumerate(self.dec_layers):
            if (
                self.training
                and self.layer_drop_rate > 0
                and random.random() < self.layer_drop_rate
            ):
                continue
            x, self_attn, cross_attn = layer(
                x,
                encoder_output,
                dec_key_padding_mask=pad_mask_dec,
                enc_key_padding_mask=pad_mask_src,
                attn_mask=causal_mask,
            )
            attn_weights[f"layer{i+1}_dec_self"] = self_attn
            attn_weights[f"layer{i+1}_dec_cross"] = cross_attn

        x = self.decoder_norm(x)
        logits = self.final_linear(x)

        return logits, attn_weights

    def forward(
        self,
        source: torch.Tensor,
        targets: torch.Tensor,
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, dict]:
        enc_out, pad_mask_src, enc_attn, ctc_input = self.encode(source, source_lengths)
        logits, dec_attn = self.decode(targets, enc_out, target_lengths, pad_mask_src)

        attn_weights = {**enc_attn, **dec_attn}
        return logits, attn_weights, ctc_input

    def score(
        self,
        prompts: torch.Tensor,
        encoder_output: torch.Tensor,
        pad_mask_src: torch.Tensor,
    ) -> torch.Tensor:
        """Score the next token logits given encoder outputs.

        This is used during recognition where decoder inputs are not padded.
        Args:
            prompts: Tensor of token ids shaped (batch, seq_len)
            encoder_output: Encoded speech features (batch, src_len, d_model)
            pad_mask_src: Padding mask for encoder outputs
        Returns:
            logits for the next token prediction shaped (batch, num_classes)
        """
        if self.training:
            raise ValueError(
                "score method is not supported during training, use forward instead"
            )

        logits, _ = self.decode(
            prompts, encoder_output, target_lengths=None, pad_mask_src=pad_mask_src
        )
        return logits[:, -1, :]
