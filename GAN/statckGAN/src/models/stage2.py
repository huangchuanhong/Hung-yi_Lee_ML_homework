import torch
import torch.nn as nn
from .ca import CA
from .utils import downsample, upsample
from .stage1 import Stage1Generator

# import sys
# sys.path.append('.')
# from ca import CA
# from utils import downsample, upsample
# from stage1 import Stage1Generator


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class Stage2Generator(nn.Module):
    def __init__(self, embeddings_dim=1024, c_dim=128, f_dim=96):
        super(Stage2Generator, self).__init__()
        self.f_dim = f_dim * 2
        self.c_dim = c_dim
        self.ca = CA(embeddings_dim, c_dim)
        self.dsample = self.downsample_()
        self.joint = nn.Sequential(
            nn.Conv2d(c_dim + self.f_dim * 4, self.f_dim * 4, 3, padding=1),
            nn.BatchNorm2d(self.f_dim * 4),
            nn.ReLU(True)
        )
        self.res_blocks1 = self.residual_blocks(self.f_dim * 4)

        self.upsample1 = upsample(self.f_dim * 4, self.f_dim * 2)
        self.upsample2 = upsample(self.f_dim * 2, self.f_dim)
        self.upsample3 = upsample(self.f_dim, self.f_dim // 2)
        self.upsample4 = upsample(self.f_dim // 2, self.f_dim // 4)
        self.conv = nn.Conv2d(self.f_dim // 4, 3, 3, padding=1)
        self.tanh = nn.Tanh()

        self.apply(init_weights)

    def downsample_(self):
        return nn.Sequential(
            nn.Conv2d(3, self.f_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            downsample(self.f_dim, self.f_dim * 2),
            downsample(self.f_dim * 2, self.f_dim * 4)
        )

    def residual_blocks(self, dim):
        return nn.Sequential(
            ResBlock(dim),
            ResBlock(dim)
        )

    def forward(self, txt_embeddings, stage1_img):
        c, mu, logvar = self.ca(txt_embeddings)
        x = self.dsample(stage1_img)
        c = c.unsqueeze(2).unsqueeze(3).repeat((1, 1, 16, 16))
        x = torch.cat((c, x), dim=1)
        x = self.joint(x)
        x = self.res_blocks1(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.conv(x)
        x = self.tanh(x)
        return x, mu, logvar

class Stage2Discriminator(nn.Module):
    def __init__(self, df_dim=96, c_dim=128):
        super(Stage2Discriminator, self).__init__()
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, df_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 128 * 128
        self.downsample2 = downsample(df_dim, df_dim * 2)   # 64 * 64
        self.downsample3 = downsample(df_dim * 2, df_dim * 4)  # 32 * 32
        self.downsample4 = downsample(df_dim * 4, df_dim * 8)  # 16 * 16
        self.downsample5 = downsample(df_dim * 8, df_dim * 16)  # 8 * 8
        self.downsample6 = downsample(df_dim * 16, df_dim * 32)  # 4*4
        self.conv0_1 = nn.Sequential(
            nn.Conv2d(df_dim * 32, df_dim * 16, 3, padding=1),
            nn.BatchNorm2d(df_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv0_2 = nn.Sequential(
            nn.Conv2d(df_dim * 16, df_dim * 8, 3, padding=1),
            nn.BatchNorm2d(df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(df_dim * 8 + c_dim, df_dim * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Conv2d(df_dim * 8, 1, 4)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, conditions, imgs):
        c = conditions.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4, 4)
        x = self.downsample1(imgs)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = self.downsample4(x)
        x = self.downsample5(x)
        x = self.downsample6(x)
        x = self.conv0_1(x)
        x = self.conv0_2(x)
        x = self.conv1(torch.cat((c, x), dim=1))
        x = self.conv2(x)
        logits = self.sigmoid(x)
        return logits.squeeze()

if __name__ == '__main__':
    s1_g = Stage1Generator()
    txt_embeddings = torch.zeros((32, 1024))
    noise = torch.zeros((32, 100)).normal_()
    x, _, _ = s1_g(txt_embeddings, noise)
    x = x.detach()
    print(f's1, x.shape={x.shape}')

    s2_g = Stage2Generator()
    x, mu, logvar = s2_g(txt_embeddings, x)
    print(f's2, x.shape={x.shape}')

    s1_d = Stage2Discriminator()
    logits = s1_d(mu, x)
    print('logits={}'.format(logits.shape))