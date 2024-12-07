import torch
from torch import nn
from torch.nn.utils import weight_norm

from src.utils.conv_utils import get_padding_to_keep_dim


class CombinedConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: list[int],
    ) -> None:
        super(CombinedConvBlock, self).__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(),
                    weight_norm(
                        nn.Conv1d(
                            in_channels=channels,
                            out_channels=channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=get_padding_to_keep_dim(dilation, kernel_size),
                        )
                    ),
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation_config: list[list[int]],
    ):
        super(ResidualBlock, self).__init__()
        self.residual_blocks = nn.ModuleList(
            [
                CombinedConvBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilations=dilations,
                )
                for dilations in dilation_config
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for residual_block in self.residual_blocks:
            temp = residual_block(x)
            x = temp + x
        return x


class MRF(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size_config: list[int],
        dilation_config: list[list[list[int]]],
    ) -> None:
        super(MRF, self).__init__()

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=channels,
                    kernel_size=kernel_size_config[i],
                    dilation_config=dilation_config[i],
                )
                for i in range(len(kernel_size_config))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = None
        k = len(self.blocks)
        for block in self.blocks:
            if res is None:
                res = block(x)
            else:
                temp = block(x)
                res = temp
        return res / k


class HiFiGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_transpose_kernels: list[int],
        upsample_rates: list[int],
        mrf_kernels: list[int],
        mrf_dilation_config: list[list[list[int]]],
    ) -> None:
        super(HiFiGenerator, self).__init__()

        self.conv_first = weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=1,
                padding=get_padding_to_keep_dim(1, 7),
            )
        )

        blocks_count = len(conv_transpose_kernels)
        blocks_in_channels: list[int] = [out_channels] + [
            out_channels // int(2**l) for l in range(1, 1 + blocks_count)
        ]

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    # TODO: Normalization?
                    nn.LeakyReLU(),
                    weight_norm(
                        nn.ConvTranspose1d(
                            in_channels=blocks_in_channels[i],
                            out_channels=blocks_in_channels[i + 1],
                            kernel_size=conv_transpose_kernels[i],
                            stride=conv_transpose_kernels[i] // 2,
                            padding=(conv_transpose_kernels[i] - upsample_rates[i])
                            // 2,
                        )
                    ),
                    MRF(
                        channels=blocks_in_channels[i + 1],
                        kernel_size_config=mrf_kernels,
                        dilation_config=mrf_dilation_config,
                    ),
                )
                for i in range(blocks_count)
            ]
        )
        self.conv_last = nn.Sequential(
            nn.LeakyReLU(),
            weight_norm(
                nn.Conv1d(
                    in_channels=blocks_in_channels[-1],
                    out_channels=1,
                    kernel_size=7,
                    dilation=1,
                    padding=get_padding_to_keep_dim(1, 7),
                )
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze()
        x = self.conv_first(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv_last(x)
        return x
