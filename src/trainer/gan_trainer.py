from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from src.utils.mel_spectrogram import MelSpectrogramConfig


class GanTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        audio_to_mel: nn.Module,
        generator_loss: nn.Module,
        discriminator_loss: nn.Module,
        metrics: nn.Module,
        generator_optimizer,
        discriminator_optimizer,
        generator_lr_scheduler,
        discriminator_lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.generator = generator
        self.discriminator = discriminator
        self.audio_to_mel = audio_to_mel
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_lr_scheduler = generator_lr_scheduler
        self.discriminator_lr_scheduler = discriminator_lr_scheduler
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint()
            raise e

    def _log_audio(self, batch: dict[str, torch.Tensor], mode: str) -> None:
        table = {}
        for i in range(min(8, batch["audio"].shape[0])):
            real_audio = batch["real_audio"][i]
            generated_audio = batch["fake_audio"][i]
            table[batch["name"][i]] = {
                "real_audio": self.writer.wandb.Audio(
                    real_audio.squeeze(0).cpu().numpy(), sample_rate=22050
                ),
                "generated_audio": self.writer.wandb.Audio(
                    generated_audio.squeeze(0).cpu().numpy(), sample_rate=22050
                ),
            }

        self.writer.add_table(
            f"audio_{mode}",
            pd.DataFrame.from_dict(
                table,
                orient="index",
            ),
        )
        self.writer.add_image(
            f"real_spectrogram_{mode}",
            plot_spectrogram(batch["real_spectrogram"][0].squeeze(0).cpu().numpy()),
        )
        self.writer.add_image(
            f"generated_spectrogram_{mode}",
            plot_spectrogram(batch["fake_spectrogram"][0].squeeze(0).cpu().numpy()),
        )

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint()

            if stop_process:  # early_stop
                break

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.generator.train()
        self.discriminator.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
                if batch_idx % 5 == 0:
                    self._log_audio(batch, "train")
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            self.train_metrics.update(
                "generator_grad_norm", self._get_grad_norm(self.generator)
            )
            self.train_metrics.update(
                "discriminator_grad_norm", self._get_grad_norm(self.discriminator)
            )

            # log current results
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Generator Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["generator_loss"]
                    )
                )
                self.logger.debug(
                    "Train Epoch: {} {} Discriminator Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["discriminator_loss"]
                    )
                )
                self.writer.add_scalar(
                    "generator learning rate",
                    self.generator_lr_scheduler.get_last_lr()[0],
                )
                self.writer.add_scalar(
                    "discriminator learning rate",
                    self.discriminator_lr_scheduler.get_last_lr()[0],
                )
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        if self.generator_lr_scheduler is not None:
            self.generator_lr_scheduler.step()

        if self.discriminator_lr_scheduler is not None:
            self.discriminator_lr_scheduler.step()

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.generator.eval()
        self.discriminator.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
                if batch_idx % 5 == 0:
                    self._log_audio(batch, "test")
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)

        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("generator_max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.generator.parameters(),
                self.config["trainer"]["generator_max_grad_norm"],
            )
        if self.config["trainer"].get("discriminator_max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.discriminator.parameters(),
                self.config["trainer"]["discriminator_max_grad_norm"],
            )

    @torch.no_grad()
    def _get_grad_norm(self, model: nn.Module, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

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
            value=0,
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

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self):
        self._save_checkpoint_part(
            "generator",
            model=self.generator,
            optimizer=self.generator_optimizer,
            lr_scheduler=self.generator_lr_scheduler,
            epoch=self._last_epoch,
            save_best=False,
        )
        self._save_checkpoint_part(
            "discriminator",
            model=self.discriminator,
            optimizer=self.discriminator_optimizer,
            lr_scheduler=self.discriminator_lr_scheduler,
            epoch=self._last_epoch,
            save_best=False,
        )

    def _save_checkpoint_part(
        self,
        model_type_suffix: str,
        model: nn.Module,
        optimizer: nn.Module,
        lr_scheduler,
        epoch: int,
        save_best=False,
        only_best=False,
    ):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(
            self.checkpoint_dir / f"{model_type_suffix}-checkpoint-epoch{epoch}.pth"
        )
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path: Path) -> None:
        self._resume_checkpoint_part(resume_path, "generator")
        self._resume_checkpoint_part(resume_path, "discriminator")

    def _resume_checkpoint_part(self, resume_path: Path, model_type: str):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = resume_path.parent / f"{model_type}-{resume_path.name}"
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"][model_type] != self.config[model_type]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        getattr(self, model_type).load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"][f"{model_type}_optimizer"]
            != self.config[f"{model_type}_optimizer"]
            or checkpoint["config"][f"{model_type}_lr_scheduler"]
            != self.config[f"{model_type}_lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            getattr(self, f"{model_type}_optimizer").load_state_dict(
                checkpoint["optimizer"]
            )
            getattr(self, f"{model_type}_lr_scheduler").load_state_dict(
                checkpoint["lr_scheduler"]
            )

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path: Path) -> None:
        self._from_pretrained_part(pretrained_path, "generator")
        self._from_pretrained_part(pretrained_path, "discriminator")

    def _from_pretrained_part(self, pretrained_path: Path, model_type: str) -> None:
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = (
            pretrained_path.parent / f"{model_type}-{pretrained_path.name}"
        )
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        getattr(self, model_type).load_state_dict(checkpoint["state_dict"])
