import torch
from torch import nn
import torch.nn.functional as F


class NonLocalModule(nn.Module):
    def __init__(self, in_channels, embed_dim=None, embed_factor=4, spatial_sub_sample=False):
        super().__init__()

        assert embed_factor >= 1
        self.embed_dim = embed_dim if embed_dim is not None else in_channels // embed_factor

        self.theta = self._conv_1x1(in_channels, self.embed_dim)
        self.phi = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)) if spatial_sub_sample else nn.Sequential(),
            self._conv_1x1(in_channels, self.embed_dim))
        self.g = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)) if spatial_sub_sample else nn.Sequential(),
            self._conv_1x1(in_channels, self.embed_dim))
        self.W = nn.Sequential(
            self._conv_1x1(self.embed_dim, in_channels),
            nn.BatchNorm2d(in_channels)
        )

        self._init_params()

    @staticmethod
    def _conv_1x1(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)

        theta = theta.view(theta.shape[:2] + (-1,))
        phi = phi.view(phi.shape[:2] + (-1,))
        g = g.view(g.shape[:2] + (-1,))

        theta_phi = torch.matmul(theta.transpose(1, 2), phi)
        attention = F.softmax(theta_phi, dim=2)

        y = torch.matmul(g, attention)
        y = y.view(y.shape[:2] + x.shape[2:])

        out = self.W(y) + x

        return out
