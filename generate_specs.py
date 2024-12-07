import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="mel_generation")
def main(config):
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    dataloaders, _ = get_dataloaders(config, device)

    model = instantiate(config.generator).to(device)

    for dataloader in dataloaders.values():
        specs_path = dataloader.dataset.spectrogram_dir

        specs_path.mkdir(parents=True, exist_ok=True)

        for batch in tqdm(dataloader):
            audio = batch["audio"].to(device)
            texts = batch["text"]

            mel_specs = model(audio=audio, texts=texts)
            for name, spec in zip(batch["name"], mel_specs):
                torch.save(spec.detach().cpu(), specs_path / f"{name}.spec")


if __name__ == "__main__":
    main()
