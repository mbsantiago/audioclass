import numpy as np
import pytest
from birdnetsnd.model import load_model
from birdnetsnd.process import process_array
from tflite_runtime.interpreter import Interpreter


@pytest.fixture
def interpreter() -> Interpreter:
    return load_model()


def test_process_array_can_process_1d_arrays(interpreter):
    array = np.zeros([144_000], dtype=np.float32)
    process_array(interpreter, array)


def test_process_array_fails_with_incorrect_size(interpreter):
    array = np.zeros([8_000], dtype=np.float32)

    with pytest.raises(ValueError, match="should consist of 144000 samples"):
        process_array(interpreter, array)


def test_process_array_fails_for_3d_array(interpreter):
    array = np.zeros([1, 1, 1], dtype=np.float32)

    with pytest.raises(ValueError):
        process_array(interpreter, array)


def test_can_process_arrays_with_dynamic_batch_size(interpreter):
    array = np.random.rand(1, 144000).astype(np.float32)
    process_array(interpreter, array)

    array = np.random.rand(2, 144000).astype(np.float32)
    process_array(interpreter, array)

    array = np.random.rand(3, 144000).astype(np.float32)
    process_array(interpreter, array)

    array = np.random.rand(1, 144000).astype(np.float32)
    process_array(interpreter, array)
