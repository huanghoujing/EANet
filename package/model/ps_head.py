import torch.nn as nn


class PartSegHead(nn.Module):
    def __init__(self, cfg):
        super(PartSegHead, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=cfg.in_c,
            out_channels=cfg.mid_c,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(cfg.mid_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=cfg.mid_c,
            out_channels=cfg.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        nn.init.normal_(self.deconv.weight, std=0.001)
        nn.init.normal_(self.conv.weight, std=0.001)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(self.deconv(x))))
        return x
