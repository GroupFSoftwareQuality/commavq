import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import torch

from utils.gpt import GPT, GPTConfig

TOKENS_PATH = os.path.join(os.path.dirname(__file__), '..', 'examples', 'tokens.npy')

# the gpt notebook prepends a BOS token to each frame row, making it 129 per frame
# (128 encode tokens + 1 BOS = tokens_per_frame from GPTConfig)
gpt_config = GPTConfig()
BOS_TOKEN = gpt_config.bos_token          # 1024
TOKENS_PER_FRAME = gpt_config.tokens_per_frame  # 129
BLOCK_SIZE = gpt_config.block_size        # 20 * 129 = 2580


# --- encode output is compatible with gpt input ---

def test_encode_token_count_per_frame_plus_bos_matches_gpt_config():
    # encode produces 128 tokens per frame, gpt expects 129 (128 + BOS)
    tokens = np.load(TOKENS_PATH)
    assert tokens.shape[1] + 1 == TOKENS_PER_FRAME

def test_encode_vocab_range_fits_gpt_vocab():
    # gpt vocab_size is 1025 (0-1024), encode tokens must all be below that
    tokens = np.load(TOKENS_PATH)
    assert tokens.max() < gpt_config.vocab_size

def test_bos_token_not_in_encode_output():
    # BOS is 1024, should never appear in raw encode tokens since vocab is 0-1023
    tokens = np.load(TOKENS_PATH)
    assert not np.any(tokens == BOS_TOKEN), "BOS token found in encode output, will collide with GPT conditioning"


# --- BOS prepending (what the gpt notebook does before feeding tokens in) ---

def test_bos_prepend_produces_correct_shape():
    tokens = np.load(TOKENS_PATH).astype(np.int32)
    tokens_with_bos = np.c_[np.ones(len(tokens), dtype=np.int32) * BOS_TOKEN, tokens]
    assert tokens_with_bos.shape[1] == TOKENS_PER_FRAME

def test_bos_is_first_column_after_prepend():
    tokens = np.load(TOKENS_PATH).astype(np.int32)
    tokens_with_bos = np.c_[np.ones(len(tokens), dtype=np.int32) * BOS_TOKEN, tokens]
    assert np.all(tokens_with_bos[:, 0] == BOS_TOKEN)

def test_original_tokens_unchanged_after_prepend():
    tokens = np.load(TOKENS_PATH).astype(np.int32)
    tokens_with_bos = np.c_[np.ones(len(tokens), dtype=np.int32) * BOS_TOKEN, tokens]
    np.testing.assert_array_equal(tokens_with_bos[:, 1:], tokens)


# --- context window truncation (gpt notebook clips to block_size) ---

def test_truncation_stays_within_block_size():
    tokens = np.load(TOKENS_PATH).astype(np.int32)
    tokens_with_bos = np.c_[np.ones(len(tokens), dtype=np.int32) * BOS_TOKEN, tokens]
    truncated = tokens_with_bos[-(BLOCK_SIZE // TOKENS_PER_FRAME - 1):].reshape(-1)
    assert truncated.shape[0] <= BLOCK_SIZE

def test_truncated_tokens_are_flat():
    tokens = np.load(TOKENS_PATH).astype(np.int32)
    tokens_with_bos = np.c_[np.ones(len(tokens), dtype=np.int32) * BOS_TOKEN, tokens]
    truncated = tokens_with_bos[-(BLOCK_SIZE // TOKENS_PER_FRAME - 1):].reshape(-1)
    assert truncated.ndim == 1

def test_truncation_preserves_token_values():
    tokens = np.load(TOKENS_PATH).astype(np.int32)
    tokens_with_bos = np.c_[np.ones(len(tokens), dtype=np.int32) * BOS_TOKEN, tokens]
    truncated = tokens_with_bos[-(BLOCK_SIZE // TOKENS_PER_FRAME - 1):].reshape(-1)
    # every value should still be a valid token (0 to vocab_size-1)
    assert truncated.min() >= 0
    assert truncated.max() < gpt_config.vocab_size


# --- tensor conversion (numpy -> torch for gpt input) ---

def test_tokens_convert_to_long_tensor():
    tokens = np.load(TOKENS_PATH).astype(np.int32)
    tokens_with_bos = np.c_[np.ones(len(tokens), dtype=np.int32) * BOS_TOKEN, tokens]
    truncated = tokens_with_bos[-(BLOCK_SIZE // TOKENS_PER_FRAME - 1):].reshape(-1)
    t = torch.tensor(truncated).to(torch.long)
    assert t.dtype == torch.long

def test_tensor_shape_is_1d():
    tokens = np.load(TOKENS_PATH).astype(np.int32)
    tokens_with_bos = np.c_[np.ones(len(tokens), dtype=np.int32) * BOS_TOKEN, tokens]
    truncated = tokens_with_bos[-(BLOCK_SIZE // TOKENS_PER_FRAME - 1):].reshape(-1)
    t = torch.tensor(truncated).to(torch.long)
    assert t.ndim == 1

def test_tensor_values_within_embedding_range():
    # GPT has an embedding table of size vocab_size, any index outside that crashes
    tokens = np.load(TOKENS_PATH).astype(np.int32)
    tokens_with_bos = np.c_[np.ones(len(tokens), dtype=np.int32) * BOS_TOKEN, tokens]
    truncated = tokens_with_bos[-(BLOCK_SIZE // TOKENS_PER_FRAME - 1):].reshape(-1)
    t = torch.tensor(truncated).to(torch.long)
    assert t.min().item() >= 0
    assert t.max().item() < gpt_config.vocab_size


# --- post generation reshaping ---

def test_generated_tokens_reshape_drops_bos_column():
    # after generation, gpt notebook reshapes back to (frames, tokens_per_frame)
    # and strips the BOS column, giving (frames, 128) to pass to the decoder
    tokens = np.load(TOKENS_PATH).astype(np.int32)
    tokens_with_bos = np.c_[np.ones(len(tokens), dtype=np.int32) * BOS_TOKEN, tokens]
    flat = tokens_with_bos.reshape(-1)
    # simulate a small generation result appended
    fake_generated = torch.zeros(TOKENS_PER_FRAME, dtype=torch.long)
    flat_t = torch.tensor(flat, dtype=torch.long)
    combined = torch.cat([flat_t, fake_generated])
    # trim to multiple of tokens_per_frame
    trimmed = combined[:(combined.shape[0] // TOKENS_PER_FRAME) * TOKENS_PER_FRAME]
    reshaped = trimmed.reshape(-1, TOKENS_PER_FRAME)
    stripped = reshaped[:, 1:].to(dtype=torch.int64)
    # should now match what encode produced originally (128 tokens per frame)
    assert stripped.shape[1] == TOKENS_PER_FRAME - 1
    assert stripped.dtype == torch.int64

def test_stripped_tokens_within_decoder_vocab():
    # after stripping BOS, all remaining tokens should be valid decoder inputs (0-1023)
    tokens = np.load(TOKENS_PATH).astype(np.int32)
    tokens_with_bos = np.c_[np.ones(len(tokens), dtype=np.int32) * BOS_TOKEN, tokens]
    flat = tokens_with_bos.reshape(-1)
    trimmed = flat[:(len(flat) // TOKENS_PER_FRAME) * TOKENS_PER_FRAME]
    reshaped = trimmed.reshape(-1, TOKENS_PER_FRAME)
    stripped = reshaped[:, 1:]
    assert stripped.min() >= 0
    assert stripped.max() < 1024  # decoder vocab size