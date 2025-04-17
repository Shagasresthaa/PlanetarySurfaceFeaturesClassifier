import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseLayer, self).__init__()
        inter_channels = bn_size * growth_rate

        # first conv: 1x1 bottleneck
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)

        # second conv: 3x3 expansion
        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        # input goes through BN-ReLU-Conv1x1 -> BN-ReLU-Conv3x3
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))

        # optional dropout regularization
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        # concatenate input with new features -> DenseNet behavior
        return torch.cat([x, new_features], 1)


# block of dense layers stacked sequentially
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            # each layer increases input channels due to concatenation
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# transition block: compresses features and downsamples
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # apply 1x1 conv -> downsample with avgpool
        x = self.conv(self.relu(self.norm(x)))
        return self.pool(x)


# Final DenseNet classifier
class AstroNet(nn.Module):
    def __init__(self, num_classes=4, growth_rate=32, block_layers=(6, 12, 24, 16), drop_rate=0.1):
        super(AstroNet, self).__init__()
        self.growth_rate = growth_rate
        num_init_features = 64

        # initial conv layer -> standard ResNet-style start
        self.conv0 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # build dense blocks and transitions
        channels = num_init_features
        self.features = nn.Sequential()
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers, channels, growth_rate, drop_rate=drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            channels += num_layers * growth_rate
            if i != len(block_layers) - 1:
                # add transition except after final block
                trans = Transition(channels, channels // 2)
                self.features.add_module(f'transition{i+1}', trans)
                channels = channels // 2

        # final BN -> global pool -> linear classifier
        self.final_bn = nn.BatchNorm2d(channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool0(self.relu0(self.norm0(self.conv0(x))))
        x = self.features(x)
        x = self.final_bn(x)
        x = self.relu0(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
