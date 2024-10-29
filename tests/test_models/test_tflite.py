import numpy as np
import pytest

pytest.importorskip("tflite_runtime")

from audioclass.models.birdnet import BirdNET
from audioclass.models.tflite import Interpreter, Signature, process_array


@pytest.fixture
def birdnet_model() -> BirdNET:
    return BirdNET.load()


@pytest.fixture
def interpreter(birdnet_model: BirdNET) -> Interpreter:
    return birdnet_model.interpreter  # type: ignore


@pytest.fixture
def signature(birdnet_model: BirdNET) -> Signature:
    return birdnet_model.signature


@pytest.mark.tflite
def test_process_array_can_process_1d_arrays(
    interpreter: Interpreter, signature: Signature
):
    array = np.zeros([144_000], dtype=np.float32)
    process_array(interpreter, signature, array)


@pytest.mark.tflite
def test_process_array_fails_with_incorrect_size(
    interpreter: Interpreter, signature: Signature
):
    array = np.zeros([8_000], dtype=np.float32)

    with pytest.raises(ValueError, match="should consist of 144000 samples"):
        process_array(interpreter, signature, array)


@pytest.mark.tflite
def test_process_array_fails_for_3d_array(
    interpreter: Interpreter, signature: Signature
):
    array = np.zeros([1, 1, 1], dtype=np.float32)

    with pytest.raises(ValueError):
        process_array(interpreter, signature, array)


@pytest.mark.tflite
def test_can_process_arrays_with_dynamic_batch_size(
    interpreter: Interpreter, signature: Signature
):
    array = np.random.rand(1, 144000).astype(np.float32)
    process_array(interpreter, signature, array)

    array = np.random.rand(2, 144000).astype(np.float32)
    process_array(interpreter, signature, array)

    array = np.random.rand(3, 144000).astype(np.float32)
    process_array(interpreter, signature, array)

    array = np.random.rand(1, 144000).astype(np.float32)
    process_array(interpreter, signature, array)
