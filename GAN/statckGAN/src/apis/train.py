import os
import torch
import torch.nn as nn
import logging
from torch.utils.data.dataloader import DataLoader
from ..datasets import Birds
from ..models import Stage1Discriminator, Stage1Generator, Stage2Discriminator, Stage2Generator
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter

def init_logger(args, stage='stage1'):
    logger = logging.Logger(__name__)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    if stage == 'stage1':
        if args.s1_log_path:
            file_handler = logging.FileHandler(args.s1_log_path, 'w')
            logger.addHandler(file_handler)
    elif stage == 'stage2':
        if args.s2_log_path:
            file_handler = logging.FileHandler(args.s2_log_path, 'w')
            logger.addHandler(file_handler)
    return logger

def discriminator_loss(discriminator, r_imgs, f_imgs, conditions, r_labels, f_labels, criterion):
    # real pairs
    r_logits = discriminator(conditions, r_imgs)
    r_loss = criterion(r_logits, r_labels)
    # wrong pairs
    w_logits = discriminator(conditions[:r_imgs.size(0)-1], r_imgs[1:])
    w_loss = criterion(w_logits, f_labels[1:])
    # fake pairs
    f_logits = discriminator(conditions, f_imgs)
    f_loss = criterion(f_logits, f_labels)
    loss = r_loss + (w_loss + f_loss) * 0.5
    return loss, r_loss, w_loss, f_loss

def kl_loss(mu, logvar):
    return (logvar.exp() - (1 + logvar) + mu ** 2).mean()


def stage1_train(args):
    logger = init_logger(args)
    if args.summary:
        summary_writer = SummaryWriter(args.s1_summary_path)
    dataset = Birds(args.data_dir, split='train', im_size=64)
    dataloader = DataLoader(dataset, batch_size=args.s1_batch_size, shuffle=True, num_workers=8, drop_last=True)
    generator = Stage1Generator(args.txt_embedding_dim, args.c_dim, args.z_dim, args.gf_dim).cuda()
    print('generator={}'.format(generator))
    discriminator = Stage1Discriminator(args.df_dim, args.c_dim).cuda()
    print('discriminator={}'.format(discriminator))
    device_ids = list(range(torch.cuda.device_count()))
    generator = DataParallel(generator, device_ids)
    discriminator = DataParallel(discriminator, device_ids)
    g_parameters = list(filter(lambda f: f.requires_grad, generator.parameters()))
    d_parameters = list(filter(lambda f: f.requires_grad, discriminator.parameters()))
    g_optimizer = torch.optim.Adam(g_parameters, args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(d_parameters, args.lr, betas=(0.5, 0.999))
    r_labels = torch.ones((args.s1_batch_size,), device='cuda:0')
    f_labels = torch.zeros((args.s1_batch_size,), device='cuda:0')
    criterion = nn.BCELoss()
    cur_lr = args.lr
    for epoch in range(args.total_epoch):
        for idx, (r_imgs, txt_embeddings) in enumerate(dataloader):
            r_imgs = r_imgs.cuda()
            txt_embeddings = txt_embeddings.cuda()
            # discriminator
            noise = torch.zeros((args.s1_batch_size, args.z_dim), device='cuda:0').normal_()
            x, mu, logvar = generator(txt_embeddings, noise)
            d_loss, r_loss, w_loss, f_loss = discriminator_loss(discriminator, r_imgs, x.detach(), mu.detach(), r_labels, f_labels, criterion)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            # generator
            noise = torch.zeros((args.s1_batch_size, args.z_dim), device='cuda:0').normal_()
            x, mu, logvar = generator(txt_embeddings, noise)
            logits = discriminator(mu.detach(), x)
            g_loss = criterion(logits, r_labels)
            kl_loss_ = kl_loss(mu, logvar)
            g_loss += kl_loss_
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            if args.summary and idx % args.summary_iters == 0 and idx > 0:
                summary_writer.add_scalar('d_loss', g_loss.item())
                summary_writer.add_scalar('r_loss', r_loss.item())
                summary_writer.add_scalar('w_loss', w_loss.item())
                summary_writer.add_scalar('f_loss', f_loss.item())
                summary_writer.add_scalar('g_loss', g_loss.item())
                summary_writer.add_scalar('kl_loss', kl_loss.item())
        if epoch % args.lr_decay_every_epoch == 0 and epoch > 0:
            logger.info(f'lr decay: {cur_lr}')
            cur_lr *= args.lr_decay_ratio
            g_optimizer = torch.optim.Adam(g_parameters, cur_lr, betas=(0.5, 0.999))
            d_optimizer = torch.optim.Adam(d_parameters, cur_lr, betas=(0.5, 0.999))
        if epoch % args.display_epoch == 0 and epoch > 0:
            logger.info(f'epoch:{epoch}, lr={cur_lr}, d_loss={d_loss}, r_loss={r_loss}, w_loss={w_loss}, f_loss={f_loss}, g_loss={g_loss}, kl_loss={kl_loss_}')
        if epoch % args.checkpoint_epoch == 0 and epoch > 0:
            if not os.path.isdir(args.s1_checkpoint_dir):
                os.makedirs(args.s1_checkpoint_dir)
            logger.info(f'saving checkpoints_{epoch}')
            torch.save(generator.state_dict(), os.path.join(args.s1_checkpoint_dir, f'generator_epoch_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(args.s1_checkpoint_dir, f'discriminator_epoch_{epoch}.pth'))
    torch.save(generator.state_dict(), os.path.join(args.s1_checkpoint_dir, 'generator.pth'))
    torch.save(generator.state_dict(), os.path.join(args.s1_checkpoint_dir, 'discriminator.pth'))
    if args.summary:
        summary_writer.close()


def stage2_train(args):
    logger = init_logger(args)
    if args.summary:
        summary_writer = SummaryWriter(args.s2_summary_path)
    dataset = Birds(args.data_dir, split='train', im_size=256)
    dataloader = DataLoader(dataset, batch_size=args.s2_batch_size, shuffle=True, num_workers=8, drop_last=True)
    generator1 = Stage1Generator(args.txt_embedding_dim, args.c_dim, args.z_dim, args.gf_dim)
    print('generator1={}'.format(generator1))
    state_dict = torch.load(args.s1_checkpoint_path)
    for n, p in generator1.state_dict().items():
        if 'module.' in state_dict:
            p.copy_(state_dict['module.' + n])
    generator1 = generator1.cuda()
    generator2 = Stage2Generator(args.txt_embedding_dim, args.c_dim, args.gf_dim).cuda()
    print(f'generator2={generator2}')
    discriminator = Stage2Discriminator(args.df_dim, args.c_dim).cuda()
    print('discriminator={}'.format(discriminator))
    device_ids = list(range(torch.cuda.device_count()))
    generator1 = DataParallel(generator1, device_ids)
    generator2 = DataParallel(generator2, device_ids)
    discriminator = DataParallel(discriminator, device_ids)
    g2_parameters = list(filter(lambda f: f.requires_grad, generator2.parameters()))
    d_parameters = list(filter(lambda f: f.requires_grad, discriminator.parameters()))
    g2_optimizer = torch.optim.Adam(g2_parameters, args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(d_parameters, args.lr, betas=(0.5, 0.999))
    r_labels = torch.ones((args.s2_batch_size,), device='cuda:0')
    f_labels = torch.zeros((args.s2_batch_size,), device='cuda:0')
    criterion = nn.BCELoss()
    cur_lr = args.lr
    generator1.eval()
    for epoch in range(args.total_epoch):
        for idx, (r_imgs, txt_embeddings) in enumerate(dataloader):
            r_imgs = r_imgs.cuda()
            txt_embeddings = txt_embeddings.cuda()
            # discriminator
            noise = torch.zeros((args.s2_batch_size, args.z_dim), device='cuda:0').normal_()
            with torch.no_grad():
                s1_img, _, _ = generator1(txt_embeddings, noise)
            s1_img = s1_img.detach()
            s2_img, mu, logvar = generator2(txt_embeddings, s1_img)
            d_loss, r_loss, w_loss, f_loss = discriminator_loss(discriminator, r_imgs, s2_img.detach(), mu.detach(), r_labels, f_labels, criterion)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            # generator
            s2_img, mu, logvar = generator2(txt_embeddings, s1_img)
            logits = discriminator(mu.detach(), s2_img)
            g_loss = criterion(logits, r_labels)
            kl_loss_ = kl_loss(mu, logvar)
            g_loss += kl_loss_
            g2_optimizer.zero_grad()
            g_loss.backward()
            g2_optimizer.step()
            if args.summary and idx % args.summary_iters == 0 and idx > 0:
                summary_writer.add_scalar('d_loss', g_loss.item())
                summary_writer.add_scalar('r_loss', r_loss.item())
                summary_writer.add_scalar('w_loss', w_loss.item())
                summary_writer.add_scalar('f_loss', f_loss.item())
                summary_writer.add_scalar('g_loss', g_loss.item())
                summary_writer.add_scalar('kl_loss', kl_loss.item())
        if epoch % args.lr_decay_every_epoch == 0 and epoch > 0:
            logger.info(f'lr decay: {cur_lr}')
            cur_lr *= args.lr_decay_ratio
            g2_optimizer = torch.optim.Adam(g2_parameters, cur_lr, betas=(0.5, 0.999))
            d_optimizer = torch.optim.Adam(d_parameters, cur_lr, betas=(0.5, 0.999))
        if epoch % args.display_epoch == 0 and epoch > 0:
            logger.info(f'epoch:{epoch}, lr={cur_lr}, d_loss={d_loss}, r_loss={r_loss}, w_loss={w_loss}, f_loss={f_loss}, g_loss={g_loss}, kl_loss={kl_loss_}')
        if epoch % args.checkpoint_epoch == 0 and epoch > 0:
            if not os.path.isdir(args.s2_checkpoint_dir):
                os.makedirs(args.s2_checkpoint_dir)
            logger.info(f'saving checkpoints_{epoch}')
            torch.save(generator2.state_dict(), os.path.join(args.s2_checkpoint_dir, f'generator_epoch_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(args.s2_checkpoint_dir, f'discriminator_epoch_{epoch}.pth'))
    torch.save(generator2.state_dict(), os.path.join(args.s2_checkpoint_dir, 'generator.pth'))
    torch.save(generator2.state_dict(), os.path.join(args.s2_checkpoint_dir, 'discriminator.pth'))
    if args.summary:
        summary_writer.close()