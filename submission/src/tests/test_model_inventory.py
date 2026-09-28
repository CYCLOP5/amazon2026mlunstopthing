"""keep tensor parameters separate from non-parameter buffers"""

import importlib.util
import json
from pathlib import Path
import struct

import pytest


spec = importlib.util.spec_from_file_location('model_inventory', Path(__file__).resolve().parents[1] / 'model_inventory.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def test_parameters_and_integer_buffers_have_separate_counts(tmp_path):
    header = json.dumps({'weight': {'dtype': 'F32', 'shape': [2, 3], 'data_offsets': [0, 24]},
                         'position_ids': {'dtype': 'I64', 'shape': [2], 'data_offsets': [24, 40]}}).encode()
    file = tmp_path / 'model.safetensors'
    file.write_bytes(struct.pack('<Q', len(header)) + header + bytes(40))
    assert inventory.tensors(file) == (6, 2)


def test_invalid_header_size_is_rejected(tmp_path):
    file = tmp_path / 'bad.safetensors'
    file.write_bytes(struct.pack('<Q', 2**40))
    with pytest.raises(ValueError, match='header length'):
        inventory.tensors(file)
