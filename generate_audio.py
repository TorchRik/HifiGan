import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from hydra.utils import instantiate
from tqdm.auto import tqdm

from src.datasets.data_utils import get_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(
    version_base=None, config_path="src/configs", config_name="audio_generation"
)
def main(config):
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    dataloaders, _ = get_dataloaders(config, device)

    text_to_mel = instantiate(config.mel_generator).to(device)
    audio_to_mel = instantiate(config.audio_to_mel).to(device)
    mel_to_audio = instantiate(config.generator).to(device)

    checkpoint = torch.load(config.checkpoint_path, device)
    mel_to_audio.load_state_dict(checkpoint["state_dict"])

    path_to_save = Path(config.path_to_save)
    path_to_save.mkdir(parents=True, exist_ok=True)
    for dataloader in dataloaders.values():
        for batch in tqdm(dataloader):
            if "audio" in batch.keys():
                spectrogram = audio_to_mel(batch["audio"].to(device))
            else:
                spectrogram = text_to_mel(batch["text"])
            spectrogram = spectrogram.to(device)

            generated_audio = mel_to_audio(spectrogram)
            for name, audio in zip(batch["name"], generated_audio):
                torchaudio.save(
                    uri=path_to_save / f"{name}.wav",
                    src=audio.reshape(1, -1).detach().cpu(),
                    sample_rate=22050,
                )


if __name__ == "__main__":
    main()
