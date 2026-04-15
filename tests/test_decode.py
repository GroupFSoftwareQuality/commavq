import numpy as np
import torch
import pytest

from unittest.mock import MagicMock, patch

# Assuming you refactor notebook code into decode.py
import decode


@pytest.fixture
def mock_decoder():
    decoder = MagicMock()
    # simulate decoder output: (1, C, H, W)
    decoder.return_value = torch.ones((1, 3, 16, 16))
    return decoder


@pytest.fixture
def dummy_tokens():
    return torch.randint(0, 10, (5,), dtype=torch.int64)


def test_load_tokens(tmp_path):
    # create dummy tokens file
    file_path = tmp_path / "tokens.npy"
    np.save(file_path, np.array([1, 2, 3]))

    tokens = decode.load_tokens(file_path)

    assert isinstance(tokens, torch.Tensor)
    assert tokens.dtype == torch.int64
    assert tokens.shape[0] == 3


def test_decode_video_shape(mock_decoder, dummy_tokens):
    decoded = decode.decode_video(mock_decoder, dummy_tokens)

    # Expect concatenated output along dim=0
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape[0] == len(dummy_tokens)


def test_decode_video_calls_decoder(mock_decoder, dummy_tokens):
    decode.decode_video(mock_decoder, dummy_tokens)

    assert mock_decoder.call_count == len(dummy_tokens)


@patch("decode.write_video")
def test_save_video(mock_write_video):
    dummy_video = np.zeros((5, 3, 16, 16))
    path = "/tmp/test.mp4"

    decode.save_video(dummy_video, path)

    mock_write_video.assert_called_once()


@patch("decode.Decoder")
@patch("decode.CompressorConfig")
def test_load_decoder(mock_config, mock_decoder_class):
    mock_instance = MagicMock()
    mock_decoder_class.return_value = mock_instance

    decoder = decode.load_decoder()

    assert decoder is not None
    mock_decoder_class.assert_called_once()