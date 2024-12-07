import typing as tp
from pathlib import Path

import pandas as pd
import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

DATA_PATH = ROOT_PATH / "data"


class LJSpeechDataset(BaseDataset):
    """
    Dataset for LJSpeech dataset.
    """

    def __init__(
        self, dataset_path: Path | str, rebuild: bool = False, *args, **kwargs
    ):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """
        self.dataset_path = (
            dataset_path if isinstance(dataset_path, Path) else Path(dataset_path)
        )
        self.transcriptions_path = self.dataset_path / "transcriptions"
        self.spectrogram_dir = self.dataset_path / "spectrogram"
        self.wavs_dir = self.dataset_path / "wavs"

        index_path = self.dataset_path / "index.json"
        if index_path.is_file() and not rebuild:
            index = read_json(index_path)
            write_json(content=index, file_path=index_path)
        else:
            index = self._create_index()

        super().__init__(index, *args, **kwargs)

    def _get_transcriptions(self) -> dict[str, str]:
        if self.transcriptions_path.is_dir():
            res = {}
            for file_path in self.transcriptions_path.iterdir():
                file_name = file_path.name
                with open(file_path, "r", encoding="utf-8") as f:
                    res[file_name] = f.read()
            return res

        metadata_path = self.dataset_path / "metadata.csv"
        if not metadata_path.is_file():
            raise ValueError(
                f"Can not find transcriptions or metadata file {metadata_path}"
            )

        data = pd.read_csv(metadata_path, names=["name", "-", "text"])

        return {
            name: text
            for name, text in zip(data["name"].values[1:], data["text"].values[1:])
        }

    def _create_index(
        self,
    ) -> list[dict[str, str]]:
        """
        Create index for the dataset. If transcriptions dir does not exist,
        algorithms tries to create it from metadata.csv. It also creates dir
        with mel-spectrogram.
        """
        transcriptions = self._get_transcriptions()

        res = []
        for file_name, text in transcriptions.items():
            wav_path = self.wavs_dir / (file_name + ".wav")
            spectrogram_path = self.spectrogram_dir / (file_name + ".spec")
            res.append(
                {
                    "name": file_name,
                    "text": text,
                    "wav_path": (wav_path if wav_path.exists() else None),
                    "spectrogram_path": (
                        spectrogram_path if spectrogram_path.exists() else None
                    ),
                }
            )
        return res

    def __getitem__(self, ind: int) -> dict[str, tp.Any]:
        data_dict = self._index[ind]

        instance_data = {
            "name": data_dict["name"],
            "text": data_dict["text"],
        }

        if data_dict["wav_path"] is not None:
            audio, sr = torchaudio.load(data_dict["wav_path"])
            instance_data["audio"] = audio
        else:
            print(data_dict)

        if data_dict["spectrogram_path"] is not None:
            spectrogram = torch.load(data_dict["spectrogram_path"])
            instance_data["spectrogram"] = spectrogram

        instance_data = self.preprocess_data(instance_data)

        return instance_data
