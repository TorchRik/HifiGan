import torch
import torch.nn.functional as F

from src.metrics.tracker import MetricTracker
from src.model import MelSpectrogramConfig
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]

        real_spectrogram = self.audio_to_mel(batch["audio"])
        fake_audio = self.generator(x=real_spectrogram)
        fake_spectrogram = self.audio_to_mel(fake_audio)

        real_spectrogram = F.pad(
            real_spectrogram,
            (
                0,
                fake_spectrogram.shape[-1] - real_spectrogram.shape[-1],
            ),
            value=MelSpectrogramConfig.pad_value,
        )

        real_audio = F.pad(
            batch["audio"],
            (
                0,
                fake_audio.shape[-1] - batch["audio"].shape[-1],
            ),
            value=MelSpectrogramConfig.pad_value,
        )

        # discriminator loss:
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.discriminator_optimizer.zero_grad()

        prediction_for_real = self.discriminator(real_audio)
        prediction_for_fake = self.discriminator(fake_audio.detach())

        discriminator_loss = self.discriminator_loss(
            prediction_for_real=prediction_for_real,
            prediction_for_fake=prediction_for_fake,
        )

        if self.is_train:
            discriminator_loss.backward()
            self._clip_grad_norm()
            self.discriminator_optimizer.step()

            self.generator_optimizer.zero_grad()

        disc_output_for_real = self.discriminator.forward_with_features_map(real_audio)
        disc_output_for_fake = self.discriminator.forward_with_features_map(fake_audio)

        generator_loss = self.generator_loss(
            generated_mel=fake_spectrogram,
            target_mel=real_spectrogram,
            disc_output_for_real=disc_output_for_real,
            disc_output_for_fake=disc_output_for_fake,
        )
        if self.is_train:
            generator_loss.backward()
            self._clip_grad_norm()
            self.generator_optimizer.step()

        batch["generator_loss"] = generator_loss.item()
        batch["discriminator_loss"] = discriminator_loss.item()
        batch["real_audio"] = real_audio.detach()
        batch["fake_audio"] = fake_audio.detach()
        batch["real_spectrogram"] = real_spectrogram.detach()
        batch["fake_spectrogram"] = fake_spectrogram.detach()

        # update metrics for each loss (in case of multiple losses)
        metrics.update("generator_loss", generator_loss.item())
        metrics.update("discriminator_loss", discriminator_loss.item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
