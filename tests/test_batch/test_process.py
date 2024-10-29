from soundevent import data

from audioclass.batch import SimpleIterator, process_iterable


def test_process_iterable_outputs_clip_predictions(
    process_array,
    file_list,
    tags,
):
    samplerate = 48000
    input_samples = 48000 * 2

    iterator = SimpleIterator.from_files(
        file_list,
        samplerate,
        input_samples,
        batch_size=4,
    )
    clip_predictions = process_iterable(
        process_array,
        iterator,
        tags,
        name="test",
    )
    assert all(isinstance(cp, data.ClipPrediction) for cp in clip_predictions)
