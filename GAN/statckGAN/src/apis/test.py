import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from ..models import Stage1Generator, Stage2Generator
from ..datasets import Birds

def stage1_test(args):
    dataset = Birds(args.data_dir, 'test', im_size=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    generator = Stage1Generator(args.txt_embedding_dim, args.c_dim, args.z_dim, args.gf_dim)
    state_dict = torch.load(args.s1_test_checkpoint_path)
    for n, p in generator.state_dict().items():
        p.copy_(state_dict['module.' + n])
    generator = generator.cuda()
    generator.eval()
    noise = torch.zeros((1, args.z_dim)).cuda()
    with torch.no_grad():
        for idx, (r_img, txt_embedding) in enumerate(dataloader):
            if idx > 10:
                break
            txt_embedding = txt_embedding.cuda()
            x, mu, logvar = generator(txt_embedding, noise)
            f_img = (0.5*x + 0.5) * 255
            f_img = f_img.cpu().numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8)
            f_img = Image.fromarray(f_img)
            if not os.path.isdir('output/test_imgs/s1'):
                os.makedirs('output/test_imgs/s1')
            f_img.save(f'output/test_imgs/s1/test_f_{idx}.jpg')
            r_img = (0.5*r_img + 0.5) * 255
            r_img = r_img.cpu().numpy().squeeze().transpose((1,2,0)).astype(np.uint8)
            r_img = Image.fromarray(r_img)
            r_img.save(f'output/test_imgs/s1/test_r_{idx}.jpg')

def stage2_test(args):
    dataset = Birds(args.data_dir, 'test', im_size=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    generator1 = Stage1Generator(args.txt_embedding_dim, args.c_dim, args.z_dim, args.gf_dim)
    generator1.load_state_dict(args.s1_test_checkpoint_path)
    generator2 = Stage2Generator(args.txt_embedding_dim, args.c_dim, args.gf_dim)
    generator2.load_stage_dict(args.s2_test_checkpoint_path)
    generator1 = generator1.cuda()
    generator2 = generator2.cuda()
    generator1.eval()
    generator2.eval()
    noise = torch.zeros((1, args.z_dim)).cuda()
    with torch.no_grad():
        for idx, (r_img, txt_embedding) in enumerate(dataloader):
            if idx > 10:
                break
            txt_embedding = txt_embedding.cuda()
            s1_img, _, _ = generator1(txt_embedding, noise)
            s2_img, _, _ = generator2(txt_embedding, s1_img)
            s1_f_img = (0.5*s1_img + 0.5) * 255
            s1_f_img = s1_f_img.cpu().numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8)
            s1_f_img = Image.fromarray(s1_f_img)
            if not os.path.isdir('output/test_imgs/s2'):
                os.makedirs('output/test_imgs/s2')
            s1_f_img.save(f'output/test_imgs/s2/test_s1_f_{idx}.jpg')
            s2_f_img = (0.5*s2_img + 0.5) * 255
            s2_f_img = s2_f_img.cpu().numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8)
            s2_f_img = Image.fromarray(s2_f_img)
            s2_f_img.save(f'output/test_imgs/s2/test_s2_f_{idx}.jpg')
            r_img = (0.5*r_img + 0.5) * 255
            r_img = r_img.cpu().numpy().squeeze().transpose((1,2,0)).astype(np.uint8)
            r_img = Image.fromarray(r_img)
            r_img.save(f'output/test_imgs/s1/test_r_{idx}.jpg')