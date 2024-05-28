from typing import Tuple

import numpy as np
from tflite_runtime.interpreter import Interpreter


def _flat_sigmoid(
    x: np.ndarray,
    sensitivity: float = -1,
    vmin: float = -15,
    vmax: float = 15,
) -> np.ndarray:
    return 1 / (1 + np.exp(sensitivity * np.clip(x, vmin, vmax)))


def process_array(
    interpreter: Interpreter,
    array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a 2D array with a TF Lite model.

    Parameters
    ----------
    interpreter
        The TF Lite model interpreter.
    array
        An array of audio data with shape (n_samples) or (batch_size, n_samples).
        The number of samples should match the input shape of the model which
        by default is 144000.

    Returns
    -------
    class_probs
        The probability of each class for each sample. This is stored
        in a 2D array with shape (batch_size, n_classes). The values are
        between 0 and 1. The value at index (i, j) is the probability that
        the class j is present in the sample i. The probability scores
        should be interpreted as the confidence of the model in the presence
        of the class in the sample. Caution should be taken when interpreting
        these values as probabilities.
    features
        The features extracted from the audio data. This is stored in a 2D
        array with shape (batch_size, n_features). The features are extracted
        from the last layer of the model and can be used for further analysis.
        The default number of features is 1024.

    Raises
    ------
    ValueError
        If the input array does not have 2 dimensions or if the number of
        samples does not match the input shape of the model.
    """
    if array.ndim == 1:
        array = array[np.newaxis, :]

    if not array.ndim == 2:
        raise ValueError("Input array must have 2 dimensions")

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if not array.shape[1] == input_details["shape"][1]:
        raise ValueError(
            "Input array should consist of {} samples".format(
                input_details["shape"][1]
            )
        )

    if array.dtype != input_details["dtype"]:
        array = array.astype(input_details["dtype"])

    if array.shape[0] != input_details["shape"][0]:
        interpreter.resize_tensor_input(
            input_details["index"],
            array.shape,
        )
        interpreter.allocate_tensors()

    interpreter.set_tensor(input_details["index"], array)

    interpreter.invoke()

    class_probs = _flat_sigmoid(
        interpreter.get_tensor(output_details["index"])
    )

    features = interpreter.get_tensor(output_details["index"] - 1)
    return class_probs, features
