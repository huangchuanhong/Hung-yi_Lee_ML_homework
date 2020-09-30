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
    discriminator = CGANDiscriminator(1, args.num_classes, args.hidden_dim, args.dim)
    if args.discriminator_load_from:
        load_checkpoint(discriminator, args.discriminator_load_from)
    discriminator = discriminator.to(args.device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    for epoch in range(args.total_epoch):
        # gen_loss, dis_loss, fdata_loss, rdata_rlabel_loss, rdata_flabel_loss
        total_losses = [0., 0., 0., 0., 0.]
        train_generator = 0
        train_discriminator = 0
        dataloader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        dataloader = iter(dataloader)
        i = 0
        show_losses = [0., 0., 0., 0., 0.]
        labels_f = torch.zeros(args.batch_size).to(args.device)
        labels_r = torch.ones(args.batch_size).to(args.device)
        while((i * args.discriminator_iters) < len(dataloader) - args.discriminator_iters):
            i += 1
            while train_discriminator < args.discriminator_iters:
                train_discriminator += 1
                data, classes_ = next(dataloader)
                data = data.to(args.device)
                fake_classes_ = (classes_ + torch.randint(1, args.num_classes, classes_.size())) % args.num_classes
                classes_ = torch.zeros((args.batch_size, args.num_classes)).scatter_(1, classes_.unsqueeze(1), 1)
                fake_classes_ = torch.zeros((args.batch_size, args.num_classes)).scatter_(1, fake_classes_.unsqueeze(1), 1)
                classes_ = classes_.to(args.device)
                fake_classes_ = fake_classes_.to(args.device)
                d_optimizer.zero_grad()
                # fake data loss
                random_inputs = torch.randn((args.batch_size, args.in_dim)).to(args.device)
                g_outputs = generator(random_inputs, classes_)
                fdata_loss = discriminator(g_outputs, classes_, labels_f)
                # real data real labels loss
                rdata_rlabel_loss = discriminator(data, classes_, labels_r)
                # real data fake label lss
                rdata_flabel_loss = discriminator(data, fake_classes_, labels_f)
                show_losses[2] += fdata_loss
                show_losses[3] += rdata_rlabel_loss
                show_losses[4] += rdata_flabel_loss
                total_losses[2] += fdata_loss
                total_losses[3] += rdata_rlabel_loss
                total_losses[4] += rdata_flabel_loss
                d_loss = (fdata_loss + rdata_flabel_loss) / 2. + rdata_rlabel_loss
                show_losses[1] += d_loss
                total_losses[1] += d_loss
                d_loss.backward()
                d_optimizer.step()
            while train_generator < args.generator_iters:
                train_generator += 1
                g_optimizer.zero_grad()
                random_inputs = torch.randn((args.batch_size, args.in_dim)).to(args.device)
                if train_generator == 1:
                    classes_ = classes_
                else:
                    classes_ = torch.randint(1, args.num_classes, (args.batch_size,))
                    classes_ = torch.zeros((args.batch_size, args.num_classes)).scatter(1, classes_, 1)
                g_outputs = generator(random_inputs, classes_)
                g_loss = discriminator(g_outputs, classes_, labels_r)
                g_loss.backward()
                g_optimizer.step()
                show_losses[0] += g_loss
                total_losses[0] += g_loss
            train_discriminator = 0
            train_generator = 0
            if i % args.display_iters == 0 and i > 0:
                logger.info('epoch{} [{}/{}]: gen_loss={}, dis_loss={}, fdata_loss={}, rdata_rlabel_loss={}, rdata_flabel_loss={}'.\
                            format(epoch, i * args.discriminator_iters, len(dataloader),
                                   show_losses[0]/(args.display_iters * args.generator_iters),
                                   show_losses[1]/(args.display_iters * args.discriminator_iters),
                                   show_losses[2]/(args.display_iters * args.discriminator_iters),
                                   show_losses[3]/(args.display_iters * args.discriminator_iters),
                                   show_losses[4]/(args.display_iters * args.discriminator_iters)))
                show_losses = [0., 0., 0., 0., 0., 0.]
            if i % args.save_iters == 0 and i > 0:
                torch.save(generator.state_dict(), args.save_dir + '/generator_epoch_{}_iter_{}.pth'.format(epoch, i * args.discriminator_iters))
                torch.save(discriminator.state_dict(), args.save_dir + '/discriminator_epoch_{}_iter_{}.pth'.format(epoch, i * args.discriminator_iters))
                logger.info('save checkpoint ' + args.save_dir + '/generator_epoch_{}_iter_{}.pth'.format(epoch, i*args.discriminator_iters))
        logger.info('epoch{}: gen_loss={}, dis_loss={}, fdata_loss={}, rdata_rlabel_loss={}, rdata_flabel_loss={}'.\
                    format(epoch,
                           total_losses[0]/(i * args.generator_iters),
                           total_losses[1]/(i * args.discriminator_iters),
                           total_losses[2]/(i * args.discriminator_iters),
                           total_losses[3]/(i * args.discriminator_iters),
                           total_losses[4]/(i * args.discriminator_iters)))


def de_transpose(x):
    x = x.transpose((1, 2, 0))
    x = (0.5*x + 0.5) * 255
    x = x.astype(np.uint8)
    return x

def test_old(args):
    generator = CGANGenerator(args.in_dim, args.num_classes, args.dim, 1)
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
    generator = CGANGenerator(args.in_dim, args.num_classes, args.dim, 1)
    state_dict = torch.load(args.test_checkpoint)
    for n, p in generator.state_dict().items():
        p.data.copy_(state_dict[n])
    generator = generator.to(args.device)
    generator.eval()
    inputs = torch.randn((size*args.num_classes, 100)).view(-1, 100).to(args.device)
    classes = np.array(range(args.num_classes)).reshape(-1, 1)
    classes = torch.from_numpy(np.tile(classes, (1, size)).reshape(-1, 1))
    classes = torch.zeros((classes.size(0), args.num_classes)).scatter_(1, classes, 1).to(args.device)
    outputs = generator(inputs, classes)

    fig, ax = plt.subplots(args.num_classes, size, figsize=(args.num_classes, size))
    for i in range(args.num_classes):
        for j in range(size):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    for i in range(args.num_classes):
        for j in range(size):
            ax[i, j].cla()
            img = outputs[size * i + j].cpu().data.numpy().transpose(1, 2, 0)
            # img = 0.5 * img + 0.5
            ax[i, j].imshow(img, cmap='gray')
    plt.savefig('test.jpg')

