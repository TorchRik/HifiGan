import torch
import torch.nn.functional as F

TENSOR_OBJECTS = ["audio", "spectrogram"]


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    if not dataset_items:
        return result_batch

    for item_name in dataset_items[0]:
        if item_name in TENSOR_OBJECTS:
            item_lengths = torch.tensor(
                [elem[item_name].shape[-1] for elem in dataset_items]
            )
            max_length = item_lengths.max().item()
            if item_name == "audio" and max_length % 256:
                max_length += 256 - max_length % 256

            result_batch[item_name] = torch.stack(
                [
                    F.pad(
                        elem[item_name],
                        (
                            0,
                            max_length - elem[item_name].shape[-1],
                        ),
                    )
                    for elem in dataset_items
                ]
            )
        else:
            result_batch[item_name] = [elem[item_name] for elem in dataset_items]

    return result_batch
