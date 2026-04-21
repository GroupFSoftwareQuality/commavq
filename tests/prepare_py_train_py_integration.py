"""
Integration test: prepare.py data is used in nanogpt's train.py get_batch function, so this file
tests with functions from prepare_py_unit_tests, then integrates with get_batch from train.py and checks 
for data integrity

"""

import os
import tempfile
import numpy as np
import torch

from prepare_py_unit_tests import process_example, write_memmap, BOS_TOKEN, EOT_TOKEN

BLOCK_SIZE = 1024
BATCH_SIZE = 4


def get_batch(path):
    """Reproduces get_batch() from train.py (CPU path)."""
    # Copied get_batch function from https://github.com/karpathy/nanoGPT/blob/master/train.py lines 116-131
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])
    del data
    return x, y, ix


def test_get_batch():
    segments = [
        # Calls process from prepare.py
        process_example(np.random.randint(0, 1024, (20, 128), dtype=np.int16))['ids']
        for _ in range(50)
    ]
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    try:
        write_memmap(segments, path)
        x, y, ix = get_batch(path)

        # Check if shapes and dtype correct
        assert x.shape == (BATCH_SIZE, BLOCK_SIZE), f"x shape mismatch: {x.shape}"
        assert y.shape == (BATCH_SIZE, BLOCK_SIZE), f"y shape mismatch: {y.shape}"
        assert x.dtype == torch.int64, f"x dtype mismatch: {x.dtype}"
        assert y.dtype == torch.int64, f"y dtype mismatch: {y.dtype}"

        # Checks if tokens are in range
        assert x.max().item() <= EOT_TOKEN, f"x contains token > EOT_TOKEN: {x.max().item()}"
        assert y.max().item() <= EOT_TOKEN, f"y contains token > EOT_TOKEN: {y.max().item()}"

        # Verify y is x offset by one using the actual sampled indices
        data = np.memmap(path, dtype=np.uint16, mode="r")
        assert data[ix[0] + 1] == y[0, 0].item(), "y is not x offset by one"

        # Check BOS and EOT tokens are present in the file
        assert BOS_TOKEN in data, "BOS_TOKEN not found"
        assert EOT_TOKEN in data, "EOT_TOKEN not found"

        del data  
    finally:
        os.unlink(path)

if __name__ == "__main__":
    test_get_batch()
    print("All tests passed")
