import torch
import torch.nn as nn
from .ca import CA
from .utils import downsample, upsample
# import sys
# sys.path.append('.')
# from ca import CA
# from utils import downsample, upsample


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

class Stage1Generator(nn.Module):
    def __init__(self, embeddings_dim=1024, c_dim=128, z_dim=100, f_dim=96):
        super(Stage1Generator, self).__init__()
        self.ca = CA(embeddings_dim, c_dim)
        self.fc_bn_relu = nn.Sequential(
            nn.Linear(c_dim + z_dim, f_dim * 16 * 4 * 4, bias=False),
            nn.BatchNorm1d(f_dim * 16 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = upsample(f_dim * 16, f_dim * 8)
        self.upsample2 = upsample(f_dim * 8, f_dim * 4)
        self.upsample3 = upsample(f_dim * 4, f_dim * 2)
        self.upsample4 = upsample(f_dim * 2, f_dim)
        self.conv = nn.Conv2d(f_dim, 3, 3, padding=1)
        self.tanh = nn.Tanh()
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.f_dim = f_dim
        self.apply(init_weights)

    def forward(self, txt_embeddings, noise):
        c, mu, logvar = self.ca(txt_embeddings)
        # print('c={}'.format(c))
        # print('mu={}'.format(mu))
        # print('logvar={}'.format(logvar))
        x = torch.cat([c, noise], dim=1)
        x = self.fc_bn_relu(x)
        x = x.reshape([-1, self.f_dim * 16, 4, 4])
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.conv(x)
        x = self.tanh(x)
        return x, mu, logvar

class Stage1Discriminator(nn.Module):
    def __init__(self, df_dim=96, c_dim=128):
        super(Stage1Discriminator, self).__init__()
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, df_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.downsample2 = downsample(df_dim, df_dim * 2)
        self.downsample3 = downsample(df_dim * 2, df_dim * 4)
        self.downsample4 = downsample(df_dim * 4, df_dim * 8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(df_dim * 8 + c_dim, df_dim * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Conv2d(df_dim * 8, 1, 4)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, conditions, imgs):
        c = conditions.unsqueeze(2).unsqueeze(3).repeat(1, 1, imgs.size(-1) // 16, imgs.size(-1) // 16)
        x = self.downsample1(imgs)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = self.downsample4(x)
        x = self.conv1(torch.cat((c, x), dim=1))
        x = self.conv2(x)
        logits = self.sigmoid(x)
        return logits.squeeze()

if __name__ == '__main__':
    s1_g = Stage1Generator()
    txt_embeddings = torch.zeros((32, 1024))
    noise = torch.zeros((32, 100)).normal_()
    x, mu, logvar = s1_g(txt_embeddings, noise)
    s1_d = Stage1Discriminator()
    logits = s1_d(mu, x)
    print('logits={}'.format(logits.shape))