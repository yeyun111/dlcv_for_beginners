import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetConvBlock(nn.Module):
    def __init__(self, input_nch, output_nch, kernel_size=3, activation=F.leaky_relu, use_bn=True, same_conv=True):
        super(UNetConvBlock, self).__init__()
        padding = kernel_size // 2 if same_conv else 0  # only support odd kernel
        self.conv0 = nn.Conv2d(input_nch, output_nch, kernel_size, padding=padding)
        self.conv1 = nn.Conv2d(output_nch, output_nch, kernel_size, padding=padding)
        self.act = activation
        self.batch_norm = nn.BatchNorm2d(output_nch) if use_bn else None

    def forward(self, x):
        x = self.conv0(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.act(x)
        x = self.conv1(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return self.act(x)


class UNet(nn.Module):
    def __init__(self, conv_channels, input_nch=3, output_nch=2, use_bn=True):
        super(UNet, self).__init__()
        self.n_stages = len(conv_channels)
        # define convolution blocks
        down_convs = []
        up_convs = []

        in_nch = input_nch
        for i, out_nch in enumerate(conv_channels):
            down_convs.append(UNetConvBlock(in_nch, out_nch, use_bn=use_bn))
            up_conv_in_ch = 2 * out_nch if i < self.n_stages - 1 else out_nch # first up conv with equal channels
            up_conv_out_ch = out_nch if i == 0 else in_nch  # last up conv with channels equal to labels
            up_convs.insert(0, UNetConvBlock(up_conv_in_ch, up_conv_out_ch, use_bn=use_bn))
            in_nch = out_nch

        self.down_convs = nn.ModuleList(down_convs)
        self.up_convs = nn.ModuleList(up_convs)

        # define output convolution
        self.out_conv = nn.Conv2d(conv_channels[0], output_nch, 1)

    def forward(self, x):
        # conv & downsampling
        down_sampled_fmaps = []
        for i in range(self.n_stages-1):
            x = self.down_convs[i](x)
            x = F.max_pool2d(x, 2, 2)
            down_sampled_fmaps.insert(0, x)

        # center convs
        x = self.down_convs[self.n_stages-1](x)
        x = self.up_convs[0](x)

        # conv & upsampling
        for i, down_sampled_fmap in enumerate(down_sampled_fmaps):
            x = torch.cat([x, down_sampled_fmap], 1)
            x = self.up_convs[i+1](x)
            x = F.upsample(x, scale_factor=2, mode='bilinear')

        x = self.out_conv(x)
        return x if self.out_conv.out_channels == 1 else F.relu(x)

