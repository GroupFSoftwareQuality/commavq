""" 
This file contains unit tests for nanogpt/prepare.py

Test functions include process(example), memmap writing loop, and HuggingFace's shards logic without having HF as a dependency.
Functions are stored in this file to eliminate dependencies so testing changes requires changing test functions. 

Test Cases/Sections: TestProcessExample, TestMemmapOutput, TestSharding, TestEdgeCases, TestReferenceRegression 

Elements tested include BOS/EOT token insertion, sharding and batching logic, and
correct datatypes and length when writing to memmap
"""
import os
import sys
import tempfile
import unittest
import numpy as np
 
# because nanoGPT datasets are flat arrays of tokens
# we use this token to separate frames
BOS_TOKEN = 1024
# we use this token to separate segments
# note that the gpt2m is only trained on tokens from the same segment and doesn't have an EOT_TOKEN
EOT_TOKEN = 1025
 
# ------ TEST FUNCTIONS ------

# def process(example) from prepare.py 
# Lines 26-34
def process_example(token_array: np.ndarray) -> dict:
    tokens = np.array(token_array)
    tokens = tokens.reshape(tokens.shape[0], -1)
    # prepend BOS_TOKEN
    tokens = np.c_[np.ones(len(tokens), dtype=np.int16)*BOS_TOKEN, tokens]
    tokens = tokens.reshape(-1)
    # append EOT_TOKEN
    tokens = np.r_[tokens, EOT_TOKEN]
    return {'ids': tokens.astype(np.int16), 'len': len(tokens.astype(np.int16))}
 
 
def write_memmap(ids_list: list[np.ndarray], out_path: str) -> int:
    # Replicates memmap writing loop from prepare.py, but checks if length is 0 and 
    # Compute total length up front so we can pre-allocate the memmap
    total_len = sum(len(a) for a in ids_list)
 
    # np.memmap with shape=(0,) can write a fake byte, so explicitly guards zero length
    if total_len == 0:
        open(out_path, 'wb').close()   # create an empty file
        return 0
 
    arr = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(total_len,))
    idx = 0
    for ids in ids_list: # Replaces huggingface sharding/batching loop to remove dependency
        chunk = np.array(ids, dtype=np.uint16)
        arr[idx: idx + len(chunk)] = chunk
        idx += len(chunk)
    arr.flush()
    return total_len
 
 
def shard_ids(ids_list: list[np.ndarray], num_shards: int) -> list[np.ndarray]:
    """
    Replicates prepare.py's `dset.shard(num_shards=..., index=..., contiguous=True)`.
    Taken from HuggingFace logic it calls, https://github.com/huggingface/datasets/blob/main/src/datasets/arrow_dataset.py#L5220 
    Lines 5281-5295
 
    """
    n = len(ids_list)
    shards = []
    for index in range(num_shards):
        div = n // num_shards
        mod = n % num_shards
        start = div * index + min(index, mod)
        end = start + div + (1 if index < mod else 0)
        shard_examples = ids_list[start:end]
        if shard_examples:
            shards.append(np.concatenate(shard_examples))
        else:
            shards.append(np.array([], dtype=np.int16))
    return shards


# ------ TEST CASES ------
 
class TestProcessExample(unittest.TestCase):
    # Tests for token processing step

    # Helper to make frame array
    def _make_frames(self, num_frames: int, tokens_per_frame: int,
                     start_val: int = 0) -> np.ndarray:
        # Builds a 2d (num_frames, tokens_per_frame) token array 
        # ie [[0, 1]
        #    [2, 3]]
        total = num_frames * tokens_per_frame
        vals  = [(start_val + i) % 1024 for i in range(total)]
        return np.array(vals, dtype=np.int16).reshape(num_frames, tokens_per_frame)
 
    # Tests BOS placement
    def test_bos_at_start_of_each_frame(self):
        # BOS_TOKEN must be first element of every frame
        frames = self._make_frames(num_frames=3, tokens_per_frame=4) 
        result = process_example(frames)
        ids = result['ids'] 
        tokens_per_frame = 4
        # Stride is number of locations in memory between start of next row
        stride = tokens_per_frame + 1  # +1 for BOS
 
        for frame_idx in range(3):
            bos_pos = frame_idx * stride # 
            self.assertEqual(
                int(ids[bos_pos]), BOS_TOKEN,
                f"Frame {frame_idx}: expected BOS at position {bos_pos}, "
                f"got {ids[bos_pos]}"
            )
 
    def test_frame_tokens_follow_bos(self):
        # Tokens in a frame need to be in same order after BOS
        # Testing with single frame with known values [10, 20, 30]
        frames = np.array([[10, 20, 30]], dtype=np.int16)
        result = process_example(frames)
        ids = result['ids']
 
        # Expected layout: [BOS, 10, 20, 30, EOT]
        expected = [BOS_TOKEN, 10, 20, 30, EOT_TOKEN]
        np.testing.assert_array_equal(
            ids.astype(np.int32), expected,
            err_msg="Single-frame token layout is wrong"
        )
 
    # Tests EOT placement
    def test_eot_at_end(self):
        # EOT token needs to be last in frame
        frames = self._make_frames(num_frames=5, tokens_per_frame=8)
        result = process_example(frames)
        self.assertEqual(
            int(result['ids'][-1]), EOT_TOKEN,
            "Last token must be EOT_TOKEN"
        )
 
    def test_exactly_one_eot(self):
        frames = self._make_frames(num_frames=5, tokens_per_frame=8)
        result = process_example(frames)
        eot_count = int(np.sum(result['ids'].astype(np.int32) == EOT_TOKEN))
        self.assertEqual(eot_count, 1, "Expected exactly 1 EOT token per example")
 
    # Tests correct length
    def test_output_length_formula(self):
        """
        Total tokens = num_frames * (tokens_per_frame + 1) + 1
                                               ↑ BOS         ↑ EOT
        """
        num_frames = 7
        tpf = 12 # Tokens per frame
        frames = self._make_frames(num_frames, tpf)
        result = process_example(frames)
        expected_len = num_frames * (tpf + 1) + 1
        self.assertEqual(
            result['len'], expected_len,
            f"Expected len={expected_len}, got {result['len']}"
        )
        self.assertEqual(
            len(result['ids']), result['len'],
            "'len' field must match actual array length"
        )
 
    def test_output_dtype_is_int16(self):
        # process_example must return int16 arrays to match cast in prepare.py
        frames = self._make_frames(2, 4)
        result = process_example(frames)
        self.assertEqual(
            result['ids'].dtype, np.int16,
            f"Expected dtype int16, got {result['ids'].dtype}"
        )
 
    # Tests multiple frames have correct format
    # [A, B] and [C, D] should make [BOS, A, B, BOS, C, D, EOT]
    def test_two_frame_layout(self):
        frames = np.array([[1, 2], [3, 4]], dtype=np.int16)
        result = process_example(frames)
        expected = np.array(
            [BOS_TOKEN, 1, 2, BOS_TOKEN, 3, 4, EOT_TOKEN],
            dtype=np.int32
        )
        np.testing.assert_array_equal(
            result['ids'].astype(np.int32), expected,
            err_msg="Two-frame layout is incorrect"
        )
 

class TestMemmapOutput(unittest.TestCase):
    # Tests memmap writing
 
    def setUp(self):
        # Each test gets its own temp directory that is cleaned automatically
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.out_path = os.path.join(self.tmp_dir.name, "test_split.bin")
 
    def tearDown(self):
        self.tmp_dir.cleanup()
 
    def test_memmap_dtype_is_uint16(self):
        ids_list = [np.array([BOS_TOKEN, 1, 2, EOT_TOKEN], dtype=np.int16)]
        total = write_memmap(ids_list, self.out_path)
 
        arr = np.memmap(self.out_path, dtype=np.uint16, mode='r')
        self.assertEqual(len(arr), total, "File length mismatch")
        # uint16 values should equal the original tokens
        self.assertEqual(int(arr[0]), BOS_TOKEN)
        self.assertEqual(int(arr[-1]), EOT_TOKEN)
 
    def test_total_length_matches_sum_of_lens(self):
        # File token count must equal sum of individual len values
        examples = [
            process_example(np.array([[i * 10 + j for j in range(5)]
                                       for i in range(3)]))
            for _ in range(8)   # 8 synthetic examples
        ]
        ids_list = [e['ids'] for e in examples]
        expected_total = sum(e['len'] for e in examples)
 
        total = write_memmap(ids_list, self.out_path)
        self.assertEqual(total, expected_total)
 
        arr = np.memmap(self.out_path, dtype=np.uint16, mode='r')
        self.assertEqual(len(arr), expected_total)
 
    # Tests if file size is double total_tokens since uint16 is 2 bytes each
    def test_file_size_in_bytes(self):
        ids_list = [np.array([BOS_TOKEN, 1, 2, EOT_TOKEN], dtype=np.int16)] * 10
        total = write_memmap(ids_list, self.out_path)
 
        file_bytes = os.path.getsize(self.out_path)
        self.assertEqual(file_bytes, total * 2,
                         "File size (bytes) must be num_tokens × 2")
 
# Tests for sharding logic that writes batches to memmap
class TestSharding(unittest.TestCase):
 
    def _make_ids_list(self, num_examples: int, tokens_per_example: int,
                       start: int = 0) -> list[np.ndarray]:
        # Create deterministic id arrays for sharding tests.
        return [
            np.arange(start + i * tokens_per_example,
                      start + (i + 1) * tokens_per_example,
                      dtype=np.int16)
            for i in range(num_examples)
        ]
 
    def test_all_tokens_appear_exactly_once(self):
        # After sharding and concatenate on all shards, tokens from sequence must appear once
        ids_list = self._make_ids_list(num_examples=20, tokens_per_example=10)
        original_flat = np.concatenate(ids_list)
 
        num_shards = 4
        shards = shard_ids(ids_list, num_shards)
        reassembled = np.concatenate(shards)
 
        np.testing.assert_array_equal(
            reassembled, original_flat,
            err_msg="Sharding dropped or duplicated tokens"
        )
 
    # Tests non-divisible dataset
    def test_sharding_non_divisible_dataset(self):
        """
        17 examples, 3 shards: 17 % 3 = 2, so HF gives first 2 shards
        6 elements each (div+1=6), last shard gets 5 (div=5).
        """
        ids_list = self._make_ids_list(num_examples=17, tokens_per_example=5)
        original_flat = np.concatenate(ids_list)

        shards = shard_ids(ids_list, num_shards=3)

        # verify HuggingFace's boundary math: first mod shards get div+1, rest get div
        div, mod = 17 // 3, 17 % 3   # div=5, mod=2
        expected_sizes = [div + (1 if i < mod else 0) for i in range(3)]  # [6, 6, 5]
        actual_sizes = [len(s) // 5 for s in shards]  # divide by tokens_per_example
        self.assertEqual(actual_sizes, expected_sizes,
                        f"Shard sizes {actual_sizes} don't match HF distribution {expected_sizes}")

        # no-drop property still holds
        reassembled = np.concatenate(shards)
        np.testing.assert_array_equal(
            reassembled, original_flat,
            err_msg="Non-divisible sharding lost or duplicated tokens"
        )
 
    # Tests shard count
    def test_correct_number_of_shards_returned(self):
        """shard_ids must always return exactly num_shards arrays."""
        # shard_ids must always return num_shards
        ids_list = self._make_ids_list(num_examples=10, tokens_per_example=4)
        for n in [1, 3, 7, 10, 20]:
            shards = shard_ids(ids_list, num_shards=n)
            self.assertEqual(len(shards), n,
                             f"Expected {n} shards, got {len(shards)}")
 
class TestEdgeCases(unittest.TestCase):
 
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.out_path = os.path.join(self.tmp_dir.name, "edge.bin")
 
    def tearDown(self):
        self.tmp_dir.cleanup()
 
    # Single frame, single token
    def test_single_frame_single_token(self):
        """Smallest possible input: 1 frame with 1 token."""
        frames = np.array([[42]], dtype=np.int16)
        result = process_example(frames)
        expected = np.array([BOS_TOKEN, 42, EOT_TOKEN], dtype=np.int32)
        np.testing.assert_array_equal(
            result['ids'].astype(np.int32), expected,
            err_msg="Single-frame single-token case failed"
        )
        self.assertEqual(result['len'], 3)
 
    # Empty ID list (no examples)
    def test_empty_dataset_writes_zero_bytes(self):
        # memmap file should have no tokens
        total = write_memmap([], self.out_path)
        self.assertEqual(total, 0)
        # File should still exist (memmap created it) with 0 bytes
        self.assertEqual(os.path.getsize(self.out_path), 0)
 
    def test_bos_and_eot_within_uint16_range(self):
        # Max uint16 is 2^16 
        self.assertLess(BOS_TOKEN, 2**16, "BOS_TOKEN overflows uint16")
        self.assertLess(EOT_TOKEN, 2**16, "EOT_TOKEN overflows uint16")
 
class TestReferenceRegression(unittest.TestCase):
    # Regression tests: known input and output, save to disk to compare against changes to process_example/process(example) in prepare.py
 
    # Hard-coded references
    # frames = [[10, 20], [30, 40]]
    # expected ids: [1024, 10, 20, 1024, 30, 40, 1025]
    REFERENCE_INPUT  = np.array([[10, 20], [30, 40]], dtype=np.int16)
    REFERENCE_OUTPUT = np.array([1024, 10, 20, 1024, 30, 40, 1025], dtype=np.int16)
 
    def test_known_input_produces_known_output(self):
        result = process_example(self.REFERENCE_INPUT)
        np.testing.assert_array_equal(
            result['ids'].astype(np.int32),
            self.REFERENCE_OUTPUT.astype(np.int32),
            err_msg="Regression: process_example output changed for known input"
        )

# ------ RUNNER ------
 
if __name__ == '__main__':
    # Each test prints PASS / FAIL / ERROR
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
 
    for cls in [
        TestProcessExample,
        TestMemmapOutput,
        TestSharding, 
        TestEdgeCases,
        TestReferenceRegression,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
 
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
 
    total   = result.testsRun
    failed  = len(result.failures) + len(result.errors)
    passed  = total - failed
    print(f"  TOTAL : {total}")
    print(f"  PASSED: {passed}  ✓")
    if failed:
        print(f"  FAILED: {failed}  ✗")
    else:
        print(f"  FAILED: 0")
 
    sys.exit(0 if result.wasSuccessful() else 1)