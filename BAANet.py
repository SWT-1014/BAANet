import torch.nn as nn
from torchvision import models
import torch


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class BAANet(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(BAANet, self).__init__()

        self.BBA_BAF = BBA_BAF()
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.layer1=mobilenet.features[0]
        self.layer2=mobilenet.features[1]
        self.layer3 = nn.Sequential(
            mobilenet.features[2],
            mobilenet.features[3],)
        self.layer4 = nn.Sequential(
            mobilenet.features[4],
            mobilenet.features[5],
            mobilenet.features[6],)
        self.layer5 = nn.Sequential(
            mobilenet.features[7],
            mobilenet.features[8],
            mobilenet.features[9],
            mobilenet.features[10],)
        self.layer6 = nn.Sequential(
            mobilenet.features[11],
            mobilenet.features[12],
            mobilenet.features[13],)
        self.layer7 = nn.Sequential(
            mobilenet.features[14],
            mobilenet.features[15],
            mobilenet.features[16], )
        self.layer8 = nn.Sequential(
            mobilenet.features[17],
            )

        self.final=nn.Sequential(
            conv_dw(24, 24, 1),
            conv_dw(24, num_classes, 1),)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        x2 = self.layer4(x1)
        l5 = self.layer5(x2)
        x3 = self.layer6(l5)
        l7 = self.layer7(x3)
        x4 = self.layer8(l7)

        cfm_feature = self.BBA_BAF(x4, x3, x2, x1)
        x = self.final(cfm_feature)

        return x



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BBA_BAF(nn.Module):
    def __init__(self):
        super(BBA_BAF, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(320, 96, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(320, 96, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(320, 32, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(96, 32, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(96, 32, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(320, 24, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(96, 24, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(32, 24, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(32, 24, 3, padding=1)

        self.conv_mul2 = BasicConv2d(96, 96, 3, padding=1)
        self.conv_mul3 = BasicConv2d(32, 32, 3, padding=1)
        self.conv_mul4 = BasicConv2d(24, 24, 3, padding=1)

        self.attention1 = attention(96)
        self.attention2 = attention(96)
        self.attention3 = attention(32)
        self.attention4 = attention(32)
        self.attention5 = attention(24)
        self.attention6 = attention(24)


    def forward(self, x1, x2, x3, x4):
        x2_1 = self.conv_mul2(self.conv_upsample1(self.upsample(x1)) * x2)
        x1_atten = self.attention1(self.conv_upsample2(self.upsample(x1)))
        x2_1_atten = self.attention2(x2_1)
        x2_out = x1_atten + x2_1_atten

        x3_1 = self.conv_mul3((self.conv_upsample3(self.upsample(self.upsample(x1))) + \
               self.conv_upsample4(self.upsample(x2)) ) * x3)
        x2_out_atten = self.attention3(self.conv_upsample5(self.upsample(x2_out)))
        x3_1_atten = self.attention4(x3_1)
        x3_out = x2_out_atten + x3_1_atten

        x4_1 = self.conv_mul4((self.conv_upsample6(self.upsample(self.upsample(self.upsample(x1)))) + \
               self.conv_upsample7(self.upsample(self.upsample(x2))) + \
               self.conv_upsample8(self.upsample(x3))) * x4)
        x3_out_atten = self.attention5(self.conv_upsample9(self.upsample(x3_out)))
        x4_1_atten = self.attention6(x4_1)
        x4_out = x3_out_atten + x4_1_atten

        return x4_out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class attention(nn.Module):
    def __init__(self, in_channel):
        super(attention, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()

    def forward(self, x):

        x0, x1 = x.chunk(2, dim=2)
        x0 = x0.chunk(2, dim=3)
        x1 = x1.chunk(2, dim=3)
        x0 = [self.ca(x0[-2]) * x0[-2], self.ca(x0[-1]) * x0[-1]]
        x0 = [self.sa(x0[-2]) * x0[-2], self.sa(x0[-1]) * x0[-1]]

        x1 = [self.ca(x1[-2]) * x1[-2], self.ca(x1[-1]) * x1[-1]]
        x1 = [self.sa(x1[-2]) * x1[-2], self.sa(x1[-1]) * x1[-1]]

        x0 = torch.cat(x0, dim=3)
        x1 = torch.cat(x1, dim=3)
        x3 = torch.cat((x0, x1), dim=2)

        x4 = self.ca(x) * x
        x4 = self.sa(x4) * x4
        x = x3 + x4
        return x





