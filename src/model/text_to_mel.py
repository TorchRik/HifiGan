import torch
from speechbrain.inference.TTS import FastSpeech2
from torch import nn

SOURCE = "speechbrain/tts-fastspeech2-ljspeech"


class TextToMelSpectrogram(nn.Module):
    def __init__(self, save_dir: str):
        super(TextToMelSpectrogram, self).__init__()

        self.fastspeech = FastSpeech2.from_hparams(
            source="speechbrain/tts-fastspeech2-ljspeech", savedir=save_dir
        )

    def forward(self, texts: list[str], **kwargs) -> torch.Tensor:
        mel_outputs, durations, pitch, energy = self.fastspeech.encode_text(texts)
        return mel_outputs
