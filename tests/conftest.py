import pytest

from hw4lib.data import ASRDataset, H4Tokenizer, LMDataset
from hw4lib.model import (
    CrossAttentionDecoderLayer,
    DecoderOnlyTransformer,
    EncoderDecoderTransformer,
    SelfAttentionDecoderLayer,
    SelfAttentionEncoderLayer,
)
from hw4lib.model.masks import CausalMask, PadMask
from hw4lib.model.positional_encoding import PositionalEncoding
from hw4lib.model.sublayers import CrossAttentionLayer, FeedForwardLayer, SelfAttentionLayer
from hw4lib.decoding.sequence_generator import SequenceGenerator


def _build_tokenizer():
    return H4Tokenizer(
        token_map={
            'char': './hw4lib/data/tokenizer_jsons/tokenizer_char.json',
            '1k': './hw4lib/data/tokenizer_jsons/tokenizer_1000.json',
            '5k': './hw4lib/data/tokenizer_jsons/tokenizer_5000.json',
            '10k': './hw4lib/data/tokenizer_jsons/tokenizer_10000.json',
        },
        token_type='char',
        validate=False,
    )


@pytest.fixture(scope="session")
def tokenizer():
    return _build_tokenizer()


@pytest.fixture(scope="function")
def decoder_layer(request):
    test_path = str(request.fspath)
    if 'selfattention' in test_path:
        return SelfAttentionDecoderLayer
    return CrossAttentionDecoderLayer


@pytest.fixture(scope="function")
def generator():
    return SequenceGenerator


@pytest.fixture(scope="function")
def encoder_layer():
    return SelfAttentionEncoderLayer


@pytest.fixture(scope="function")
def mask_gen_fn(request):
    test_path = str(request.fspath)
    if 'mask_causal' in test_path:
        return CausalMask
    return PadMask


@pytest.fixture(scope="function")
def positional_encoding_fn():
    return PositionalEncoding


@pytest.fixture(scope="function")
def cross_attention():
    return CrossAttentionLayer


@pytest.fixture(scope="function")
def feedforward():
    return FeedForwardLayer


@pytest.fixture(scope="function")
def self_attn():
    return SelfAttentionLayer


@pytest.fixture(scope="function")
def transformer(request):
    test_path = str(request.fspath)
    if 'encoder_decoder' in test_path:
        return EncoderDecoderTransformer
    return DecoderOnlyTransformer


@pytest.fixture(scope="function")
def dataset(request, tokenizer):
    # Route fixture to the appropriate dataset type based on the requesting test module
    if 'test_dataset_lm' in str(request.fspath):
        lm_config = {
            'root': './hw4_data_subset/hw4p1_data',
            'subset': 0.01,
            'batch_size': 4,
            'NUM_WORKERS': 0,
        }
        return LMDataset(partition='train', config=lm_config, tokenizer=tokenizer)

    data_config = {
        'root': './hw4_data_subset/hw4p2_data',
        'subset': 0.01,
        'batch_size': 4,
        'NUM_WORKERS': 0,
        'num_feats': 80,
        'norm': 'global_mvn',
        'specaug': False,
        'specaug_conf': {
            'apply_time_mask': False,
            'apply_freq_mask': False,
            'num_freq_mask': 0,
            'num_time_mask': 0,
            'freq_mask_width_range': 0,
            'time_mask_width_range': 0,
        },
    }

    return ASRDataset(
        partition='train-clean-100',
        config=data_config,
        tokenizer=tokenizer,
        isTrainPartition=True,
        global_stats=None,
    )
