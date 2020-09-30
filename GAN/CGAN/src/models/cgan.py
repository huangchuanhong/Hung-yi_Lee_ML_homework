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

class LossModel(nn.Module):
    def __init__(self, feats_dim, num_classes, hidden_dim):#feats, classes, labels):
        super(LossModel, self).__init__()
        self.fc1 = nn.Linear(feats_dim + num_classes, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, feats, classes, labels):
        x = self.fc1(torch.cat((feats, classes), dim=1))
        x = self.lrelu(self.bn1(x))
        x = self.sigmoid(self.fc2(x))
        loss = self.bce_loss(x, labels)
        return loss

class GenFeatsModel(nn.Module):
    def __init__(self, in_dim, dim=64):
        super(GenFeatsModel, self).__init__()
        self.conv0 = nn.Conv2d(in_dim, dim, 5, 2, 2)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv_bn_relu2 = ConvBNLReLU(dim, 2 * dim)
        self.conv_bn_relu3 = ConvBNLReLU(2 * dim, 4 * dim)
        self.conv_bn_relu4 = ConvBNLReLU(4 * dim, 8 * dim)
        self.conv = nn.Conv2d(8 * dim, dim, 4)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, inputs):
        x = self.conv0(inputs)
        x = self.lrelu(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv_bn_relu4(x)
        x = self.lrelu(self.bn(self.conv(x)))
        return x.squeeze()


class Discriminator(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim, dim=64):
        super(Discriminator, self).__init__()
        self.gen_feats_model = GenFeatsModel(in_dim, dim)
        self.loss_model = LossModel(dim, num_classes, hidden_dim)

    def forward(self, inputs, classes, labels):
        x = self.gen_feats_model(inputs)
        loss = self.loss_model(x, classes, labels)
        return loss


if __name__ == '__main__':
    g = Generator()
    z = torch.randn((10, 100))
    c = torch.randint(0, 10, (10, 1))
    c = torch.zeros((10, 10)).scatter_(1, c, 1)
    x = g(z, c)
    print(x.shape)
    d = Discriminator(3, 10, 64, 64)
    labels = torch.zeros((10,))
    y = d(x, c, labels)
    print(y)

