import torch.nn as nn
import torch
import random
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding
from .decoder_layers import SelfAttentionDecoderLayer, CrossAttentionDecoderLayer
from .encoder_layers import SelfAttentionEncoderLayer
from .speech_embedding import SpeechEmbedding
import warnings
from torchinfo import summary

'''
This file contains two key transformer architectures:

1. DecoderOnlyTransformer: Used for language modeling tasks (like GPT)
2. EncoderDecoderTransformer: Used for ASR (Automatic Speech Recognition) tasks
'''

## -------------------------------------------------------------------------------------------------
## Decoder-Only Transformer
## -------------------------------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    '''
    A Pre-LN Decoder-Only Transformer model.
    '''
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
        super().__init__()
        
        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len         = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes     = num_classes
        self.num_layers      = num_layers
        
        # Decoder layers
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

        # Embedding + PE + head
        self.target_embedding    = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout             = nn.Dropout(dropout)
        self.norm                = nn.LayerNorm(d_model)
        self.final_linear        = nn.Linear(d_model, num_classes)

        # Weight tying
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def forward(
        self,
        padded_targets: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        padded_targets: (B, T)
        target_lengths: (B,)
        returns:
            seq_out: (B, T, num_classes)
            running_att: dict of attention maps
        """
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")

        # Padding mask (True at PAD positions)
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_targets, target_lengths)

        # Causal mask (prevent attending to future)
        causal_mask = CausalMask(padded_targets)

        # Embedding + PE + dropout
        x = self.target_embedding(padded_targets)      # (B, T, d_model)
        x = self.positional_encoding(x)               # (B, T, d_model)
        x = self.dropout(x)

        running_att = {}
        for i in range(self.num_layers):
            # LayerDrop
            if (
                self.training
                and self.layer_drop_rate > 0.0
                and random.random() < self.layer_drop_rate
            ):
                continue

            x, att = self.dec_layers[i](
                x,
                key_padding_mask=pad_mask_dec,
                attn_mask=causal_mask,
            )
            running_att[f"layer{i+1}_dec_self"] = att

        # Final norm + projection
        x = self.norm(x)
        seq_out = self.final_linear(x)
        return seq_out, running_att

    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        """
        batch_prompts: (B, T) no padding
        returns logits for next token: (B, num_classes)
        """
        if self.training:
            raise ValueError(
                "score method is not supported during training, use forward instead"
            )
        seq_out, _ = self.forward(batch_prompts, target_lengths=None)
        return seq_out[:, -1, :]

    

## -------------------------------------------------------------------------------------------------
## Encoder-Decoder Transformer
## -------------------------------------------------------------------------------------------------
class EncoderDecoderTransformer(nn.Module):
    '''
    A Pre-LN Encoder-Decoder Transformer model for ASR tasks.
    '''
    def __init__(
            self,
            input_dim: int,  
            time_reduction: int, 
            reduction_method: Literal['lstm', 'conv', 'both'], 
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
            skip_encoder_pe: bool = False,
            skip_decoder_pe: bool = False,
    ):
        '''
        Initialize the Encoder-Decoder Transformer model.
        '''
        super().__init__()

        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.skip_encoder_pe = skip_encoder_pe
        self.skip_decoder_pe = skip_decoder_pe

        # Encoder layers
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

        # Decoder layers
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

        # Source embedding: NOTE output_dim, not d_model kwarg
        self.source_embedding = SpeechEmbedding(
            input_dim=input_dim,
            output_dim=d_model,
            time_reduction=time_reduction,
            reduction_method=reduction_method,
            dropout=dropout,
        )

        # Target embedding + PE + head
        self.target_embedding    = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout             = nn.Dropout(dropout)
        self.encoder_norm        = nn.LayerNorm(d_model)
        self.decoder_norm        = nn.LayerNorm(d_model)
        self.final_linear        = nn.Linear(d_model, num_classes)

        # CTC head
        self.ctc_head            = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Weight tying if enabled
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def encode(self, padded_sources, source_lengths):
        x_enc, x_enc_lengths = self.source_embedding(padded_sources, source_lengths)

        if not self.skip_encoder_pe:
            x_enc = self.positional_encoding(x_enc)
        x_enc = self.dropout(x_enc)

        pad_mask_src = PadMask(x_enc, x_enc_lengths)

        running_att = {}
        for i in range(self.num_encoder_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            x_enc, attention = self.enc_layers[i](x_enc, key_padding_mask=pad_mask_src)
            running_att[f"layer{i+1}_enc_self"] = attention

        x_enc = self.encoder_norm(x_enc)

        ctc_logits = self.ctc_head(x_enc)          # (B, T_enc, V)
        ctc_log_probs = ctc_logits.transpose(0, 1) # (T_enc, B, V)

        ctc_inputs = {
            "log_probs": ctc_log_probs,
            "lengths": x_enc_lengths,
        }
        return x_enc, pad_mask_src, running_att, ctc_inputs

    def decode(
        self, 
        padded_targets: torch.Tensor, 
        encoder_output: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
        pad_mask_src: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        '''
        Decode the target sequence conditioned on the encoder output.
        '''
        pad_mask_tgt = None
        if target_lengths is not None:
            pad_mask_tgt = PadMask(padded_targets, target_lengths)

        if pad_mask_tgt is None and self.training:
            warnings.warn(
                "pad_mask_tgt is None, unless you are using the decoder as a "
                "standalone model or doing inference, you should provide target_lengths"
            )

        causal_mask = CausalMask(padded_targets)

        # Embedding + (optional) PE + dropout
        x_dec = self.target_embedding(padded_targets)
        if not self.skip_decoder_pe:
            x_dec = self.positional_encoding(x_dec)
        x_dec = self.dropout(x_dec)

        running_att = {}
        for i in range(self.num_decoder_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue

            x_dec, self_attn, cross_attn = self.dec_layers[i](
                x_dec,
                encoder_output,
                dec_key_padding_mask=pad_mask_tgt,
                enc_key_padding_mask=pad_mask_src,
                attn_mask=causal_mask,
            )
            running_att[f'layer{i+1}_dec_self'] = self_attn
            running_att[f'layer{i+1}_dec_cross'] = cross_attn

        x_dec = self.decoder_norm(x_dec)
        seq_out = self.final_linear(x_dec)

        return seq_out, running_att

    def forward(
        self,
        padded_sources: torch.Tensor,
        padded_targets: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict, dict]:
        """
        Forward pass for the encoder-decoder transformer.
        """
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")
        if self.training and source_lengths is None:
            raise ValueError("source_lengths must be provided during training")

        # 1) Encode
        encoder_output, pad_mask_src, enc_running_att, ctc_inputs = self.encode(
            padded_sources,
            source_lengths,
        )

        # 2) Decode
        seq_out, dec_running_att = self.decode(
            padded_targets,
            encoder_output,
            target_lengths=target_lengths,
            pad_mask_src=pad_mask_src,
        )

        # 3) Merge attention
        running_att = {**enc_running_att, **dec_running_att}

        return seq_out, running_att, ctc_inputs


    def score(
        self,
        batch_prompts: torch.Tensor,
        encoder_output: torch.Tensor,
        pad_mask_src: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Score the next token for given encoder output and prompt.
        '''
        if self.training:
            raise ValueError("score method is not supported during training")

        seq_out, _ = self.decode(batch_prompts, encoder_output, None, pad_mask_src)
        return seq_out[:, -1, :]


    @classmethod
    def from_pretrained_decoder(
        cls,
        decoder_checkpoint_path: str,
        config: dict,
    ) -> Tuple['EncoderDecoderTransformer', dict]:
        """
        Initialize an encoder-decoder transformer with decoder weights
        from a pretrained decoder-only model.
        """
        print("\n=== Initializing Encoder-Decoder from Pretrained Decoder ===")
        print(f"Loading checkpoint from: {decoder_checkpoint_path}")
        
        # Create new encoder-decoder model
        print("\nCreating new encoder-decoder model...")
        model = cls(**config)

        # Load decoder checkpoint
        print("Loading pretrained decoder weights...")
        checkpoint = torch.load(decoder_checkpoint_path, map_location='cpu', weights_only=True)
        decoder_state_dict = checkpoint['model_state_dict']
        
        # Track named parameters
        transferred_params = []
        new_params = []
        
        def transfer_module_weights(target_module, prefix):
            module_state_dict = {
                k.replace(prefix, ''): v 
                for k, v in decoder_state_dict.items()
                if k.startswith(prefix)
            }
            if not module_state_dict:
                return
            param_count = sum(p.numel() for p in target_module.parameters())
            print(f"  - Transferring {prefix} ({param_count:,} parameters)")
            target_module.load_state_dict(module_state_dict)
            for name, param in target_module.named_parameters():
                transferred_params.append((f"{prefix}{name}", param))

        # Transfer shared components
        print("\nTransferring shared components:")
        transfer_module_weights(model.target_embedding, 'target_embedding.')
        transfer_module_weights(model.final_linear, 'final_linear.')
        transfer_module_weights(model.decoder_norm, 'norm.')
        
        # Transfer decoder layers
        num_layers = min(
            len({k.split('.')[1] for k in decoder_state_dict.keys() if k.startswith('dec_layers.')}),
            model.num_decoder_layers,
        )
        print(f"\nTransferring decoder layers (up to {num_layers} layers):")
        
        for i in range(num_layers):
            print(f"\nLayer {i + 1}/{num_layers}:")
            transfer_module_weights(
                model.dec_layers[i].self_attn,
                f'dec_layers.{i}.self_attn.'
            )
            transfer_module_weights(
                model.dec_layers[i].ffn,
                f'dec_layers.{i}.ffn.'
            )
        
        # Collect new parameters with their names
        print("\nCollecting new parameters...")
        for name, param in model.named_parameters():
            is_new = True
            for transferred_name, transferred_param in transferred_params:
                if param is transferred_param:
                    is_new = False
                    break
            if is_new:
                new_params.append((name, param))
        
        print("\n=== Initialization Complete ===")
        return model, {'transferred': transferred_params, 'new': new_params}

    def log_param_groups(self, param_groups: list) -> None:
        """Log information about parameter groups."""
        print("\nParameter groups:")
        total_params = 0
        total_trainable = 0
        
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            trainable = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_params += num_params
            total_trainable += trainable
            
            print(f"\n{group['name']}:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Trainable: {trainable:,}")
            print(f"  LR factor: {group.get('lr_factor', 1.0)}")
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total trainable: {total_trainable:,}")


## -------------------------------------------------------------------------------------------------
## Test Cases (optional)
## -------------------------------------------------------------------------------------------------

def get_decoder_only_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths


def get_encoder_decoder_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths


def test_decoder_only(
    num_layers: int = 12,
    num_heads: int = 8,
    d_model: int = 512,
    d_ff: int = 2048,
    dropout: float = 0.1,
    max_len: int = 300,
    num_classes: int = 1000,
):
    padded_targets, target_lengths = get_decoder_only_inputs(max_len, num_classes)
    model = DecoderOnlyTransformer(num_layers, d_model, num_heads, d_ff, dropout, max_len, num_classes)
    summary(model, input_data=[padded_targets, target_lengths])


if __name__ == "__main__":
    test_decoder_only()

