import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models.resnet import conv3x3


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

        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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
            x = self.max_pooling(x)
            down_sampled_fmaps.insert(0, x)

        # center convs
        x = self.down_convs[self.n_stages-1](x)
        x = self.up_convs[0](x)

        # conv & upsampling
        for i, down_sampled_fmap in enumerate(down_sampled_fmaps):
            x = torch.cat([x, down_sampled_fmap], 1)
            x = self.up_convs[i+1](x)
            x = F.upsample(x, scale_factor=2, mode='bilinear')

        return self.out_conv(x)
        #x = self.out_conv(x)
        #return x if self.out_conv.out_channels == 1 else F.relu(x)


class BasicResBlock(nn.Module):

    def __init__(self, input_nch, output_nch, groups=1):
        super(BasicResBlock, self).__init__()
        self.transform_conv = nn.Conv2d(input_nch, output_nch, 1)
        self.bn1 = nn.BatchNorm2d(output_nch)
        self.conv1 = nn.Conv2d(output_nch, output_nch, 3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(output_nch)
        self.conv2 = nn.Conv2d(output_nch, output_nch, 3, padding=1, groups=groups, bias=False)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transform_conv(x)
        residual = x

        out = self.bn1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)

        out += residual

        return out


class TriangleNet(nn.Module):
    def __init__(self, conv_channels, input_nch, output_nch, groups=1):
        super(TriangleNet, self).__init__()
        self.input_nch = input_nch
        self.output_nch = output_nch
        self.pyramid_height = len(conv_channels)

        blocks = [list() for _ in range(self.pyramid_height)]
        for i in range(self.pyramid_height):
            for j in range(i, self.pyramid_height):
                if i == 0 and j == 0:
                    blocks[i].append(BasicResBlock(input_nch, conv_channels[j], groups=groups))
                else:
                    blocks[i].append(BasicResBlock(conv_channels[j-1], conv_channels[j], groups=groups))

        for i in range(self.pyramid_height):
            blocks[i] = nn.ModuleList(blocks[i])
        self.blocks = nn.ModuleList(blocks)

        self.down_sample = nn.MaxPool2d(3, 2, 1)
        self.up_samples = nn.ModuleList([nn.Upsample(scale_factor=2**i, mode='bilinear') for i in range(1, self.pyramid_height)])

        self.channel_out_convs = nn.ModuleList([nn.Conv2d(conv_channels[-1], output_nch, 1) for _ in range(self.pyramid_height)])
        self.out_conv = nn.Conv2d(self.pyramid_height * conv_channels[-1], output_nch, 1)

    def forward(self, x):
        # forward & expand
        x = [self.blocks[0][0](x)]
        for i in range(1, self.pyramid_height):
            x.append(self.down_sample(x[-1]))
            for j in range(i+1):
                x[j] = self.blocks[j][i-j](x[j])

        # upsampling & conv
        if self.training:
            ms_out = [self.channel_out_convs[i](x[i]) for i in range(self.pyramid_height)]
        x = [x[0]] + [self.up_samples[i-1](x[i]) for i in range(1, self.pyramid_height)]

        # final 1x1 conv
        out = self.out_conv(torch.cat(x, 1))
        return [out] + ms_out if self.training else out


class PSPTriangleNet(nn.Module):
    def __init__(self, conv_channels, input_nch, output_nch, groups):
        super(PSPTriangleNet, self).__init__()
        self.input_nch = input_nch
        self.output_nch = output_nch
        self.pyramid_height = len(conv_channels)

        blocks = []
        for i in range(self.pyramid_height-1):
            if i == 0:
                blocks.append(BasicResBlock(input_nch, conv_channels[i], groups=groups))
            else:
                blocks.append(BasicResBlock(conv_channels[i-1], conv_channels[i], groups=groups))

        ms_blocks = []
        for i in range(self.pyramid_height):
            ms_blocks.append(BasicResBlock(conv_channels[-2], conv_channels[-1]//self.pyramid_height))
        self.blocks = nn.ModuleList(blocks)
        self.ms_blocks = nn.ModuleList(ms_blocks)

        self.down_samples = nn.ModuleList([nn.MaxPool2d(2**i+1, 2**i, 2**(i-1)) for i in range(1, self.pyramid_height)])
        self.up_samples = nn.ModuleList([nn.Upsample(scale_factor=2**i, mode='bilinear') for i in range(1, self.pyramid_height)])

        self.channel_out_convs = nn.ModuleList([nn.Conv2d(conv_channels[-1]//self.pyramid_height, output_nch, 1) for _ in range(self.pyramid_height)])
        self.out_conv = nn.Conv2d(conv_channels[-1], output_nch, 1)

    def forward(self, x):
        # forward & expand
        for i in range(self.pyramid_height-1):
            x = self.blocks[i](x)
        x = [self.ms_blocks[0](x)] + [self.down_samples[i](self.ms_blocks[i](x)) for i in range(self.pyramid_height-1)]

        # upsampling & conv
        if self.training:
            ms_out = [self.channel_out_convs[i](x[i]) for i in range(self.pyramid_height)]
        x = [x[0]] + [self.up_samples[i-1](x[i]) for i in range(1, self.pyramid_height)]

        # final 1x1 conv
        out = self.out_conv(torch.cat(x, 1))
        return [out] + ms_out if self.training else out

