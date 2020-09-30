import torch
import torch.nn as nn
import torch.nn.functional as F

def DConvBNReLU(in_channel, out_channel):
    return nn.Sequential(
        # nn.ConvTranspose2d(in_channel, out_channel, kernel_size=5, stride=2, padding=2, bias=False, output_padding=1),
        nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )

def ConvBNLReLU(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=2, padding=2, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(0.2)
        # nn.ReLU()
    )

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, in_dim=100, num_classes=10, dim=64, out_dim=3):
        super(Generator, self).__init__()
        self.fc_bn_relu = nn.Sequential(
            nn.Linear(in_dim + num_classes, 4*4*dim*8, bias=False),
            nn.BatchNorm1d(4*4*dim*8),
            nn.ReLU()
        )
        self.dconv_bn_relu1 = DConvBNReLU(dim*8, dim*4)
        self.dconv_bn_relu2 = DConvBNReLU(dim*4, dim*2)
        self.dconv_bn_relu3 = DConvBNReLU(dim*2, dim)
        self.dconv = nn.ConvTranspose2d(dim, out_dim, 5, 2, 2, output_padding=1)
        self.tanh = nn.Tanh()
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.dim = dim
        self.apply(weight_init)

    def forward(self, inputs, classes):
        inputs = torch.cat((inputs, classes), dim=1)
        x = self.fc_bn_relu(inputs).view((-1, self.dim*8, 4, 4))
        x = self.dconv_bn_relu1(x)
        x = self.dconv_bn_relu2(x)
        x = self.dconv_bn_relu3(x)
        x = self.dconv(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        # self.conv_bn_relu1 = ConvBNLReLU(in_dim, dim)
        self.conv0 = nn.Conv2d(in_dim, dim, 5, 2, 2)
        self.lrelu = nn.LeakyReLU(0.2)
        # self.lrelu = nn.ReLU()
        self.conv_bn_relu2 = ConvBNLReLU(dim, 2*dim)
        self.conv_bn_relu3 = ConvBNLReLU(2*dim, 4*dim)
        self.conv_bn_relu4 = ConvBNLReLU(4*dim, 8*dim)
        self.conv = nn.Conv2d(8*dim, 1, 4)
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):
        # x = self.conv_bn_relu1(inputs)
        x = self.conv0(inputs)
        x = self.lrelu(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv_bn_relu4(x)
        x = self.conv(x)
        x = self.sigmoid(x).view(-1)
        return x


if __name__ == '__main__':
    g = Generator(100, 64)
    for m in g._modules:
        print(m)
    #inputs = torch.randn((2, 100))
    #x = g(inputs)
    #print(x.shape)

    #d = Discriminator(3, 64)
    #x = d(x)
    #print(x.shape)

