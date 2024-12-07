import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    """
    Loss gor HiFi Gan generatore
    """

    @staticmethod
    def forward(
        prediction_for_real: list[torch.Tensor],
        prediction_for_fake: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0

        for real_prediction, fake_prediction in zip(
            prediction_for_real, prediction_for_fake
        ):
            loss += torch.mean((1 - real_prediction) ** 2) + torch.mean(
                fake_prediction**2
            )
        return loss
