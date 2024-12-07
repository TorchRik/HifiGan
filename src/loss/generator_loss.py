import torch
import torch.nn.functional as F
from torch import nn


class GeneratorLoss(nn.Module):
    """
    Loss gor HiFi Gan generator
    """

    def __init__(
        self,
        mell_loss_weight: float = 45.0,
        features_loss_weight: float = 2.0,
        disc_loss_weight: float = 1.0,
    ):
        super(GeneratorLoss, self).__init__()
        self.mell_loss_weight = mell_loss_weight
        self.features_loss_weight = features_loss_weight
        self.disc_loss_weight = disc_loss_weight

    @staticmethod
    def _get_discriminator_loss(
        disc_predictions_fake: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0
        for disc_prediction in disc_predictions_fake:
            loss += torch.mean((1 - disc_prediction) ** 2)
        return loss

    @staticmethod
    def _get_features_loss(
        disc_features_real: list[list[torch.Tensor]],
        disc_features_fake: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        loss = 0
        for real_features_list, fake_features_list in zip(
            disc_features_real, disc_features_fake
        ):
            for real_feature, fake_feature in zip(
                real_features_list, fake_features_list
            ):
                loss += F.l1_loss(real_feature, fake_feature)
        return loss

    def forward(
        self,
        generated_mel: torch.Tensor,
        target_mel: torch.Tensor,
        disc_output_for_real: tuple[list[torch.tensor], list[list[torch.Tensor]]],
        disc_output_for_fake: tuple[list[torch.tensor], list[list[torch.Tensor]]],
    ):
        disc_predictions_real, disc_features_real = disc_output_for_real
        disc_predictions_fake, disc_features_fake = disc_output_for_fake

        mell_loss = F.l1_loss(generated_mel, target_mel)
        disc_loss = self._get_discriminator_loss(disc_predictions_fake)
        feature_loss = self._get_features_loss(
            disc_features_real=disc_features_real, disc_features_fake=disc_features_fake
        )

        return (
            mell_loss * self.mell_loss_weight
            + disc_loss * self.disc_loss_weight
            + feature_loss * self.features_loss_weight
        )
