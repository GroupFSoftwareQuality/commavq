import sys
sys.path.append('..')

import numpy as np
import torch

from utils.gpt import GPT, GPTConfig
from utils.vqvae import Decoder, CompressorConfig


def load_model():
    config = GPTConfig()
    with torch.device('meta'):
        model = GPT(config)
    model.load_state_dict_from_url(
        'https://huggingface.co/commaai/commavq-gpt2m/resolve/main/pytorch_model.bin',
        assign=True
    )
    return model.eval().to(device='cpu', dtype=torch.bfloat16), config


def load_tokens(config):
    tokens = np.load("../examples/tokens.npy").astype(np.int32)
    tokens = np.c_[np.ones(len(tokens), dtype=np.int32)*config.bos_token, tokens]
    tokens = tokens[-(config.block_size//config.tokens_per_frame - 1):].reshape(-1)
    return torch.tensor(tokens, device='cpu')


def test_conditioning():
    model, config = load_model()
    tokens_condition = load_tokens(config)

    assert tokens_condition.ndim == 1
    assert (tokens_condition == config.bos_token).any()
    assert tokens_condition.shape[0] <= config.block_size

    print("TEST 1 (Conditioning) PASSED")


def test_generation():
    model, config = load_model()
    tokens_condition = load_tokens(config)

    test_tokens = model.generate(
        tokens_condition[-(config.block_size-config.tokens_per_frame):],
        config.tokens_per_frame
    )

    assert test_tokens.shape[0] == config.tokens_per_frame

    print("TEST 2 (Generation) PASSED")


def test_reshaping():
    model, config = load_model()
    tokens_condition = load_tokens(config)

    tokens_condition = tokens_condition.reshape(-1, config.tokens_per_frame)
    tokens_condition = tokens_condition[:, 1:].to(dtype=torch.int64)

    assert tokens_condition.shape[1] == config.tokens_per_frame - 1
    assert tokens_condition.dtype == torch.int64

    print("TEST 3 (Reshaping) PASSED")


def test_decoder():
    config = CompressorConfig()
    with torch.device('meta'):
        decoder = Decoder(config)

    decoder.load_state_dict_from_url(
        'https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder_pytorch_model.bin',
        assign=True
    )

    decoder = decoder.eval().to(device='cpu')

    gpt_config = GPTConfig()
    tokens_condition = load_tokens(gpt_config)
    tokens_condition = tokens_condition.reshape(-1, gpt_config.tokens_per_frame)
    tokens_condition = tokens_condition[:, 1:].to(dtype=torch.int64)

    with torch.no_grad():
        decoded = decoder(tokens_condition[0][None])

    assert len(decoded.shape) == 4

    print("TEST 4 (Decoder) PASSED")


if __name__ == "__main__":
    test_conditioning()
    test_generation()
    test_reshaping()
    test_decoder()