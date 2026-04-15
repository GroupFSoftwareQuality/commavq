import numpy as np
import torch
from utils.video import write_video, transpose_and_clip
from utils.vqvae import Decoder, CompressorConfig


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_decoder():
    config = CompressorConfig()
    with torch.device("meta"):
        decoder = Decoder(config)

    decoder.load_state_dict_from_url(
        "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder_pytorch_model.bin",
        assign=True,
    )
    return decoder.eval().to(device=_get_device())


def load_tokens(path):
    tokens = np.load(path).astype(np.int64)
    return torch.from_numpy(tokens).to(device=_get_device())


def decode_video(decoder, tokens):
    decoded_video = []

    with torch.no_grad():
        for i in range(len(tokens)):
            decoded = decoder(tokens[i][None])
            decoded_video.append(decoded)

    decoded_video = torch.cat(decoded_video, dim=0).cpu().numpy()
    return transpose_and_clip(decoded_video)


def save_video(video, path):
    write_video(video, path, fps=20)
