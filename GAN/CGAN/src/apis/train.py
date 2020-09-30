import torch
import torch.nn as nn
import logging
import numpy as np
import cv2
from torch.utils.data.dataloader import DataLoader
from ..models import CGANDiscriminator, CGANGenerator, LossModel
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def init_logger(args):
    logger = logging.Logger(__name__)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path, 'w')
        logger.addHandler(file_handler)
    return logger

def load_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    for n, p in model.named_parameters():
        p.data.copy_(state_dict[n])

def train(args):
    logger = init_logger(args)
    transform = transforms.Compose(
            [transforms.Resize((64, 64)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5], std=[0.5])]
        )
    dataset = datasets.MNIST('data', train=True, transform=transform, download=True)
    generator = CGANGenerator(args.in_dim, args.num_classes, args.dim, 1)
    if args.generator_load_from:
        load_checkpoint(generator, args.generator_load_from)
    generator = generator.to(args.device)
    discriminator = CGANDiscriminator(1, args.dim)
    if args.discriminator_load_from:
        load_checkpoint(discriminator, args.discriminator_load_from)
    f_loss_model = LossModel()
    discriminator = discriminator.to(args.device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss().to(args.device)
    for epoch in range(args.total_epoch):
        total_gen_loss = 0.
        total_dis_loss = 0.
        train_generator = 0
        train_discriminator = 0
        dataloader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        dataloader = iter(dataloader)
        i = 0
        gen_loss = 0.
        dis_loss = 0.
        while((i * args.discriminator_iters) < len(dataloader) - args.discriminator_iters):
            i += 1
            while train_discriminator < args.discriminator_iters:
                train_discriminator += 1
                data, classes_ = next(dataloader).to(args.device)
                classes_ = torch.zeros((args.batch_size, args.num_classes)).scatter_(1, classes_, 1)
                d_optimizer.zero_grad()
                # generator.eval()
                # discriminator.train()
                # with torch.no_grad():
                random_inputs = torch.randn((args.batch_size, args.in_dim)).to(args.device)
                g_outputs = generator(random_inputs, classes_)
                d_outputs_f = discriminator(g_outputs)
                f_loss =
                labels_f = torch.zeros(args.batch_size).to(args.device)
                loss_f = criterion(d_outputs_f, labels_f)
                d_outputs_r = discriminator(data)
                labels_r = torch.ones(args.batch_size).to(args.device)
                loss_r = criterion(d_outputs_r, labels_r)
                loss = (loss_r + loss_f) / 2.
                loss.backward()
                d_optimizer.step()
                total_dis_loss += loss
                dis_loss += loss
            while train_generator < args.generator_iters:
                train_generator += 1
                g_optimizer.zero_grad()
                # generator.train()
                # discriminator.eval()
                random_inputs = torch.randn((args.batch_size, args.in_dim)).to(args.device)
                g_outputs = generator(random_inputs)
                # with torch.no_grad():
                d_outputs = discriminator(g_outputs)
                # print('gen, d_outputs={}'.format(d_outputs))
                labels = torch.ones(args.batch_size).to(args.device)
                loss = criterion(d_outputs, labels)
                # print('generator loss ={}'.format(loss))
                loss.backward()
                g_optimizer.step()
                total_gen_loss += loss
                gen_loss += loss
            train_discriminator = 0
            train_generator = 0
            if i % args.display_iters == 0 and i > 0:
                logger.info('epoch{} [{}/{}]: gen_loss={}, dis_loss={}'.format(epoch, i * args.discriminator_iters, len(dataloader),
                                                                            gen_loss/(args.display_iters * args.generator_iters),
                                                                            dis_loss/(args.display_iters * args.discriminator_iters)))
                gen_loss = 0.
                dis_loss = 0.
            if i % args.save_iters == 0 and i > 0:
                torch.save(generator.state_dict(), args.save_dir + '/generator_epoch_{}_iter_{}.pth'.format(epoch, i * args.discriminator_iters))
                torch.save(discriminator.state_dict(), args.save_dir + '/discriminator_epoch_{}_iter_{}.pth'.format(epoch, i * args.discriminator_iters))
                logger.info('save checkpoint ' + args.save_dir + '/generator_epoch_{}_iter_{}.pth'.format(epoch, i*args.discriminator_iters))
        logger.info('epoch{}: gen_loss={}, dis_loss={}'.format(epoch, total_gen_loss/(i * args.generator_iters),
                                                            total_dis_loss/(i * args.discriminator_iters)))


def de_transpose(x):
    x = x.transpose((1, 2, 0))
    x = (0.5*x + 0.5) * 255
    x = x.astype(np.uint8)
    return x

def test_old(args):
    generator = DCGANGenerator(args.in_dim, args.dim)
    state_dict = torch.load(args.test_checkpoint)
    for n, p in generator.state_dict().items():
        p.data.copy_(state_dict[n])
    generator = generator.to(args.device)
    generator.eval()
    random_data = torch.randn((1, args.in_dim)).to(args.device)
    g_outputs = generator(random_data).detach().cpu().numpy()
    g_outputs = g_outputs.squeeze(0)#view((3, args.dim, args.dim))
    g_outputs = de_transpose(g_outputs)
    print(g_outputs)
    cv2.imwrite('test.jpg', g_outputs)

def test(args):
    size = 10
    generator = DCGANGenerator(args.in_dim, args.dim)
    state_dict = torch.load(args.test_checkpoint)
    for n, p in generator.state_dict().items():
        p.data.copy_(state_dict[n])
    generator = generator.to(args.device)
    generator.eval()
    inputs = torch.randn((size*size, 100)).view(-1, 100).to(args.device)
    outputs = generator(inputs)
 
    size = 10
    fig, ax = plt.subplots(size, size, figsize=(size, size))
    for i in range(size):
        for j in range(size):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    for i in range(size):
        for j in range(size):
            ax[i, j].cla()
            img = outputs[size*i + j].cpu().data.numpy().transpose(1,2,0)
            img = 0.5 * img + 0.5
            ax[i, j].imshow(img)
    #fig.text(0.5, 0.04, ha='center')
    plt.savefig('test.jpg')
