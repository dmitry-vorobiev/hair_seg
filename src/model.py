import torch
import torch.nn.functional as F
import torchvision

from torch import nn, Tensor
from torchvision.models.mobilenet import ConvBNReLU


def conv_block(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        ConvBNReLU(in_channels, in_channels, kernel_size, stride, groups=in_channels),
        ConvBNReLU(in_channels, out_channels, 1, stride=1))


def dec_block(in_channels, out_channels, kernel_size=3):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, 
                  groups=in_channels, bias=False),
        nn.Conv2d(in_channels, out_channels, 1, bias=True),
        nn.ReLU6(inplace=True))


class SkipLayer(nn.Module):
    def __init__(self, in_channels, out_channels, upscale=2):
        super(SkipLayer, self).__init__()
        self.up = nn.Upsample(scale_factor=upscale) if upscale > 1 else None
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.up is not None:
            x = self.up(x)
        return x + self.conv(y)


class HairMatteNet(nn.Module):
    def __init__(self, img_channels=3, num_classes=2):
        super(HairMatteNet, self).__init__()
        enc_channels = [32, 64, 128, 128, 256, 256] + [512] * 6 + [1024, 1024]
        strides = [1 + i % 2 for i in range(6)] + [1] * 5 + [2, 1]
        
        encoder = [ConvBNReLU(img_channels, enc_channels[0], stride=2)]
        encoder += [conv_block(enc_channels[i], enc_channels[i+1], stride=strides[i])
                    for i in range(len(enc_channels) - 1)]
        
        self.encoder = nn.Sequential(*encoder)
        self.skips = [None] * 4
        
        skip_idxs = [1, 3, 5, 11]
        for i, j in enumerate(skip_idxs):
            cache = self.create_cache_hook(i)
            self.encoder[j].register_forward_hook(cache)
        
        skip_channels = map(enc_channels.__getitem__, reversed(skip_idxs))
        dec_channels = [1024] + [64] * 4
        decoder = []
        for i, skip_ch in enumerate(skip_channels):
            in_ch, out_ch = dec_channels[i], dec_channels[i+1]
            decoder += [SkipLayer(skip_ch, in_ch), dec_block(in_ch, out_ch)]
            
        decoder += [nn.Upsample(scale_factor=2, mode="nearest"), 
                    dec_block(64, 64),
                    nn.Conv2d(64, num_classes, 1, bias=True), 
                    nn.Softmax2d()]
        
        self.decoder = nn.ModuleList(decoder)
        
    def create_cache_hook(self, idx: int):
        def _hook(m: nn.Module, inp: Tensor, out: Tensor):
            # saving in reverse order
            self.skips[-1-idx] = out
        return _hook
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        
        for i, y in enumerate(self.skips):
            up_add, conv = self.decoder[i*2: (i+1)*2]
            x = up_add(x, y)
            self.skips[i] = None
            x = conv(x)
            
        for i in range(len(self.skips) * 2, len(self.decoder)):
            x = self.decoder[i](x)
        
        return x
