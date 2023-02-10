import torch.nn as nn


class Convolution(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(num_features=channels)
        self.activation = activation

    def forward(self, x):
        x = x.transpose(1, 2)  # (Batch, Dim, Time)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (Batch, 2*Dim, Time)
        x = nn.functional.glu(x, dim=1)  # (Batch, Dim, Time)

        # 1D depth-wise convolution
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)
