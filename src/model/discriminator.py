import typing as tp

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm

from src.utils.conv_utils import get_padding_to_keep_dim


class MPDBlock(nn.Module):
    def __init__(
        self,
        period: int,
    ) -> None:
        super(MPDBlock, self).__init__()
        self.period = period

        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    weight_norm(
                        nn.Conv2d(
                            in_channels=1 if l == 1 else int(2 ** (4 + l)),
                            out_channels=int(2 ** (5 + l)),
                            kernel_size=(5, 1),
                            stride=(3, 1),
                            dilation=1,
                            padding=(get_padding_to_keep_dim(1, 5), 0),
                        )
                    ),
                    nn.LeakyReLU(),
                )
                for l in range(1, 5)
            ]
        )
        self.conv_blocks.append(
            nn.Sequential(
                weight_norm(
                    nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=(5, 1),
                        dilation=1,
                        padding=(get_padding_to_keep_dim(1, 5), 0),
                    )
                ),
                nn.LeakyReLU(),
            )
        )
        self.conv_blocks.append(
            weight_norm(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1,
                    kernel_size=(3, 1),
                    padding=(get_padding_to_keep_dim(1, 3), 0),
                )
            )
        )

    def _reshape_x(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape, channels_shape, time_shape = x.shape
        if time_shape % self.period != 0:
            pad_count = self.period - (time_shape % self.period)
            x = F.pad(x, (0, pad_count), "reflect")

        return x.view(
            batch_shape, channels_shape, x.shape[-1] // self.period, self.period
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = self._reshape_x(x)
        for block in self.conv_blocks:
            x = block(x)

        return x.flatten(1, -1)

    def forward_with_features_map(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self._reshape_x(x)
        features_maps = []
        for block in self.conv_blocks:
            x = block(x)
            features_maps.append(x)
        return x.flatten(1, -1), features_maps


class MPD(nn.Module):
    def __init__(self, periods: list[int]) -> None:
        super(MPD, self).__init__()
        self.mdp_blocks = nn.ModuleList([MPDBlock(period) for period in periods])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [mpd_block(x) for mpd_block in self.mdp_blocks]

    def forward_with_features_maps(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        res = []
        features_maps: list[list[torch.Tensor]] = []
        for block in self.mdp_blocks:
            block_res, block_feature_maps = block.forward_with_features_map(x)
            res.append(block_res)
            features_maps.append(block_feature_maps)
        return res, features_maps


class MSDBlock(nn.Module):
    def __init__(
        self,
        norm_function: tp.Callable = weight_norm,
    ):
        super(MSDBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    norm_function(
                        nn.Conv1d(
                            in_channels=1,
                            out_channels=16,
                            kernel_size=15,
                            stride=1,
                            dilation=1,
                            padding=get_padding_to_keep_dim(1, 15),
                        )
                    ),
                    nn.LeakyReLU(),
                )
            ]
        )
        for i in range(4):
            self.layers.append(
                nn.Sequential(
                    norm_function(
                        nn.Conv1d(
                            in_channels=int(16 * (4 ** min(3, i))),
                            out_channels=int(16 * (4 ** min(3, i + 1))),
                            kernel_size=41,
                            stride=4,
                            groups=int(4 ** (i + 1)),
                            padding=get_padding_to_keep_dim(1, 41),
                        )
                    ),
                    nn.LeakyReLU(),
                )
            )
        self.layers.append(
            nn.Sequential(
                norm_function(
                    nn.Conv1d(
                        in_channels=1024,
                        out_channels=1024,
                        kernel_size=5,
                        stride=1,
                        dilation=1,
                        padding=get_padding_to_keep_dim(1, 5),
                    )
                ),
                nn.LeakyReLU(),
            ),
        )
        self.layers.append(
            norm_function(
                nn.Conv1d(
                    in_channels=1024,
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    padding=get_padding_to_keep_dim(1, 3),
                )
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = block(x)
        return x

    def forward_with_features_map(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features_maps = []
        for block in self.layers:
            x = block(x)
            features_maps.append(x)
        return x, features_maps


class MSD(nn.Module):
    def __init__(self) -> None:
        super(MSD, self).__init__()
        self.msp_blocks = nn.ModuleList(
            [
                MSDBlock(norm_function=spectral_norm),
                MSDBlock(norm_function=weight_norm),
                MSDBlock(norm_function=weight_norm),
            ]
        )
        self.poolings = nn.ModuleList(
            [
                nn.Identity(),
                nn.AvgPool1d(4, 2, padding=2),
                nn.AvgPool1d(4, 2, padding=2),
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        res = []
        for pooling, mps_block in zip(self.poolings, self.msp_blocks):
            x = pooling(x)
            res.append(mps_block(x))
        return res

    def forward_with_features_maps(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        res = []
        features_maps: list[list[torch.Tensor]] = []
        for pooling, mps_block in zip(self.poolings, self.msp_blocks):
            x = pooling(x)
            block_res, block_feature_maps = mps_block.forward_with_features_map(x)
            res.append(block_res)
            features_maps.append(block_feature_maps)
        return res, features_maps


class HiFiDiscriminator(nn.Module):
    def __init__(self, mpd_periods: list[int]) -> None:
        super(HiFiDiscriminator, self).__init__()
        self.mpd = MPD(periods=mpd_periods)
        self.msd = MSD()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self.mpd(x)
        prediction.extend(self.msd(x))
        return prediction

    def forward_with_features_map(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        res, feature_maps = self.msd.forward_with_features_maps(x)
        msd_res, msd_feature_maps = self.msd.forward_with_features_maps(x)

        res.extend(msd_res)
        feature_maps.extend(msd_feature_maps)
        return res, feature_maps
