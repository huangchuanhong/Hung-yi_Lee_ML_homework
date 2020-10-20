import torch.nn as nn

def downsample(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )

def upsample(in_dim, out_dim):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True)
    )