import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from utils.gpt import GPT, GPTConfig
from utils.vqvae import Decoder, CompressorConfig


def test_full_pipeline():
    # Load GPT
    gpt_config = GPTConfig()
    with torch.device('meta'):
        model = GPT(gpt_config)

    model.load_state_dict_from_url(
        'https://huggingface.co/commaai/commavq-gpt2m/resolve/main/pytorch_model.bin',
        assign=True
    )

    model = model.eval().to(device='cpu', dtype=torch.bfloat16)

    # Load tokens
    tokens = np.load("examples/tokens.npy").astype(np.int32)
    tokens = np.c_[np.ones(len(tokens), dtype=np.int32)*gpt_config.bos_token, tokens]
    tokens = tokens[-(gpt_config.block_size//gpt_config.tokens_per_frame - 1):].reshape(-1)
    tokens = torch.tensor(tokens, device='cpu')
    tokens = tokens.to(torch.long)

    # Generate small amount
    new_tokens = model.generate(tokens[-50:], 10)
    tokens = torch.cat([tokens, new_tokens], axis=0)

    # Reshape
    tokens = tokens[:(tokens.shape[0] // gpt_config.tokens_per_frame) * gpt_config.tokens_per_frame]
    tokens = tokens.reshape(-1, gpt_config.tokens_per_frame)
    tokens = tokens[:, 1:].to(dtype=torch.int64)

    # Load decoder
    comp_config = CompressorConfig()
    with torch.device('meta'):
        decoder = Decoder(comp_config)

    decoder.load_state_dict_from_url(
        'https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder_pytorch_model.bin',
        assign=True
    )

    decoder = decoder.eval().to(device='cpu')

    # Decode one frame
    with torch.no_grad():
        decoded = decoder(tokens[0][None])

    # Assertions
    assert tokens.ndim == 2
    assert tokens.shape[1] == gpt_config.tokens_per_frame - 1
    assert len(decoded.shape) == 4

    print("INTEGRATION TEST PASSED")


if __name__ == "__main__":
    test_full_pipeline()