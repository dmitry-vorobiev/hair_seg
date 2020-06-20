import torch
import torch.nn.functional as F
import torchvision

from torch import nn, Tensor
from torchvision.models.mobilenet import ConvBNReLU, InvertedResidual


class EncoderBlock(nn.Sequential):
    """Depthwise separable convolution from the original MobileNet architecture.

    MobileNet paper:
    https://arxiv.org/pdf/1704.04861.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(EncoderBlock, self).__init__(
            ConvBNReLU(in_channels, in_channels, kernel_size, stride, groups=in_channels),
            ConvBNReLU(in_channels, out_channels, 1, stride=1))


class DecoderLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        padding = kernel_size // 2
        super(DecoderLayer, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU6(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, up=2):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=up)
        self.skip = nn.Conv2d(skip_channels, in_channels, 1, bias=False)  # add bias ???
        self.conv = DecoderLayer(in_channels, out_channels)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # yellow (upsample) block on scheme
        x = self.up(x) + self.skip(y)
        return self.conv(x)  # orange block


class HairMatteNet(nn.Module):
    """
    Implementation of HairMatteNet from `Real-time deep hair matting on mobile devices` paper
    https://arxiv.org/pdf/1712.07168.pdf
    """
    def __init__(self, img_channels=3, num_classes=2):
        super(HairMatteNet, self).__init__()
        enc_channels = [32, 64, 128, 128, 256, 256] + [512] * 6 + [1024, 1024]
        strides = [1, 2] * 3 + [1] * 5 + [2, 1]  # 13 layers

        encoder = [ConvBNReLU(img_channels, enc_channels[0], stride=2)]
        encoder += [EncoderBlock(enc_channels[i], enc_channels[i + 1], stride=strides[i])
                    for i in range(len(enc_channels) - 1)]
        self.encoder = nn.Sequential(*encoder)

        skip_indices = [1, 3, 5, 11]
        self.skips = []
        for i in skip_indices:
            self.encoder[i].register_forward_hook(self.save_act)

        skip_channels = map(enc_channels.__getitem__, reversed(skip_indices))
        dec_channels = [1024] + [64] * 4

        decoder = [DecoderBlock(dec_channels[i], dec_channels[i + 1], skip_ch)
                   for i, skip_ch in enumerate(skip_channels)]
        self.decoder_main = nn.ModuleList(decoder)

        self.decoder_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            DecoderLayer(64, 64),
            nn.Conv2d(64, num_classes, 1, bias=True),
            nn.Softmax2d())

    def save_act(self, m: nn.Module, inp: Tensor, out: Tensor):
        self.skips.append(out)

    def forward(self, x: Tensor) -> Tensor:
        # to ensure we don't have any tensor cached from previous
        # forward pass (in case if it was interrupted)
        self.skips = []

        x = self.encoder(x)

        for layer in self.decoder_main:
            x = layer(x, self.skips.pop())

        return self.decoder_out(x)


class HairMatte_MobileNetV2(nn.Module):
    """
    HairMatteNet with MobileNetV2 backbone
    """
    def __init__(self, pretrained=False, num_classes=2):
        super(HairMatte_MobileNetV2, self).__init__()
        mnet_v2 = torchvision.models.mobilenet_v2(pretrained=pretrained)
        # skip last layer with 320 -> 1280 channels
        self.encoder = mnet_v2.features[:-1]

        skip_indices = [1, 3, 6, 13]
        self.skips = []
        for i in skip_indices:
            self.encoder[i].register_forward_hook(self.save_act)

        skip_channels = [self.encoder[i].conv[-1].num_features
                         for i in reversed(skip_indices)]
        dec_channels = [320] + [64] * 4

        decoder = [DecoderBlock(dec_channels[i], dec_channels[i + 1], skip_ch)
                   for i, skip_ch in enumerate(skip_channels)]
        self.decoder_main = nn.ModuleList(decoder)

        self.decoder_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            DecoderLayer(64, 64),
            nn.Conv2d(64, num_classes, 1, bias=True),
            nn.Softmax2d())

        self.init_parameters(init_encoder=not pretrained)

    def init_parameters(self, init_encoder=False):
        if init_encoder:
            for m in self.encoder.modules():
                if isinstance(m, ConvBNReLU):
                    nn.init.kaiming_normal_(m[0].weight, a=0)
                elif isinstance(m, InvertedResidual):
                    # conv layer before final BN (no ReLU)
                    nn.init.kaiming_normal_(m.conv[-2].weight, a=1)
                elif isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        for decoder in [self.decoder_main, self.decoder_out]:
            for m in decoder.modules():
                if isinstance(m, DecoderLayer):
                    # there are no ReLU after the first conv2D
                    nn.init.kaiming_normal_(m[0].weight, a=1)
                    nn.init.kaiming_normal_(m[1].weight, a=0)
                elif isinstance(m, DecoderBlock):
                    nn.init.kaiming_normal_(m.skip.weight, a=1)
                elif isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def save_act(self, m: nn.Module, inp: Tensor, out: Tensor):
        self.skips.append(out)

    def forward(self, x: Tensor) -> Tensor:
        # to ensure we don't have any tensor cached from previous
        # forward pass (in case if it was interrupted)
        self.skips = []

        x = self.encoder(x)

        for layer in self.decoder_main:
            x = layer(x, self.skips.pop())

        return self.decoder_out(x)
