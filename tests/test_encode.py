import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import tempfile
import torch

TOKENS_PATH = os.path.join(os.path.dirname(__file__), '..', 'examples', 'tokens.npy')
VOCAB_SIZE = 1024  # from CompressorConfig


# --- token range validation (using the real tokens.npy from the repo) ---

def test_tokens_file_exists():
    assert os.path.exists(TOKENS_PATH), "tokens.npy not found in examples/"

def test_all_tokens_within_vocab_range():
    tokens = np.load(TOKENS_PATH)
    assert tokens.min() >= 0, f"found negative token value: {tokens.min()}"
    assert tokens.max() < VOCAB_SIZE, f"token {tokens.max()} exceeds vocab size {VOCAB_SIZE}"

def test_no_nan_or_inf_in_tokens():
    tokens = np.load(TOKENS_PATH).astype(np.float32)
    assert not np.any(np.isnan(tokens)), "NaN found in tokens"
    assert not np.any(np.isinf(tokens)), "Inf found in tokens"


# --- output shape ---

def test_token_array_is_2d():
    # encode.ipynb produces (num_frames, tokens_per_frame)
    tokens = np.load(TOKENS_PATH)
    assert tokens.ndim == 2, f"expected 2D array, got shape {tokens.shape}"

def test_tokens_per_frame_matches_expected():
    # actual tokens.npy from the repo has 128 tokens per frame
    tokens = np.load(TOKENS_PATH)
    assert tokens.shape[1] == 128, f"expected 128 tokens per frame, got {tokens.shape[1]}"


# --- save and reload ---

def test_save_reload_roundtrip():
    tokens = np.load(TOKENS_PATH)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'tokens_copy.npy')
        np.save(path, tokens)
        reloaded = np.load(path)
    np.testing.assert_array_equal(tokens, reloaded)

def test_reload_dtype_unchanged():
    tokens = np.load(TOKENS_PATH)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'tokens_copy.npy')
        np.save(path, tokens)
        reloaded = np.load(path)
    assert reloaded.dtype == tokens.dtype

def test_reload_shape_unchanged():
    tokens = np.load(TOKENS_PATH)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'tokens_copy.npy')
        np.save(path, tokens)
        reloaded = np.load(path)
    assert reloaded.shape == tokens.shape


# --- frame preprocessing (transform_img logic) ---

def test_transform_img_output_shape():
    from utils.video import transform_img
    # OUTPUT_SIZE is (256, 128) -> transform_img returns (H, W, 3)
    fake_frame = np.zeros((874, 1164, 3), dtype=np.uint8)
    result = transform_img(fake_frame)
    assert result.shape == (128, 256, 3), f"unexpected output shape: {result.shape}"

def test_transform_img_output_dtype():
    from utils.video import transform_img
    fake_frame = np.zeros((874, 1164, 3), dtype=np.uint8)
    result = transform_img(fake_frame)
    assert result.dtype == np.uint8

def test_torch_permute_produces_correct_shape():
    # simulate what encode.ipynb does after transform_img
    # frames shape after transform: (N, H, W, C) -> permute -> (N, C, H, W)
    N, H, W, C = 5, 128, 256, 3
    fake_frames = np.zeros((N, H, W, C), dtype=np.uint8)
    t = torch.from_numpy(fake_frames).permute(0, 3, 1, 2).float()
    assert t.shape == (N, C, H, W), f"unexpected shape after permute: {t.shape}"


# --- encoder config ---

def test_compressor_config_defaults():
    from utils.vqvae import CompressorConfig
    config = CompressorConfig()
    assert config.vocab_size == 1024
    assert config.resolution == 256
    assert config.in_channels == 3

def test_quantized_resolution():
    from utils.vqvae import CompressorConfig
    config = CompressorConfig()
    # resolution 256 // 2^(5-1) = 16
    assert config.quantized_resolution == 16


# --- cpu fallback ---

def test_tokens_loadable_without_gpu():
    # the saved .npy file should always be loadable regardless of device
    tokens = np.load(TOKENS_PATH)
    t = torch.from_numpy(tokens.astype(np.int64))
    assert t.device.type == 'cpu'

def test_encoder_init_on_cpu():
    from utils.vqvae import Encoder, CompressorConfig
    config = CompressorConfig()
    # should construct without error even without a GPU
    encoder = Encoder(config)
    assert encoder is not None


# --- single frame edge case ---

def test_single_frame_token_shape():
    tokens = np.load(TOKENS_PATH)
    single = tokens[:1]
    assert single.shape == (1, 128)

def test_single_frame_save_reload():
    tokens = np.load(TOKENS_PATH)
    single = tokens[:1]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'single.npy')
        np.save(path, single)
        reloaded = np.load(path)
    np.testing.assert_array_equal(single, reloaded)
