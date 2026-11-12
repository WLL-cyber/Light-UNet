import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=bias
        )

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = DepthwiseSeparableConv(out_size, out_size, kernel_size=3, padding=1)
        self.up    = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], dim=1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


# UNet 
class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [16, 32, 64, 128, 128]
        else:
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [64, 256, 512, 1024, 2048]

        self.up4 = unetUp(in_filters[4] + in_filters[3], in_filters[3])
        self.up3 = unetUp(in_filters[3] + in_filters[2], in_filters[2])
        self.up2 = unetUp(in_filters[2] + in_filters[1], in_filters[1])
        self.up1 = unetUp(in_filters[1] + in_filters[0], in_filters[0])
        self.final = nn.Conv2d(in_filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        if hasattr(self, 'vgg'):

            f1, f2, f3, f4, f5 = self.vgg(x)
        else:

            f1, f2, f3, f4, f5 = self.resnet(x)
        

        d4 = self.up4(f4, f5)
        d3 = self.up3(f3, d4)
        d2 = self.up2(f2, d3)
        d1 = self.up1(f1, d2)
        out = self.final(d1)
        return out

if __name__ == "__main__":

    model = Unet(num_classes=2, pretrained=False, backbone='vgg')
    x = torch.randn(1, 3, 1024, 1024)
    output = model(x)
    print("output:", output.shape)