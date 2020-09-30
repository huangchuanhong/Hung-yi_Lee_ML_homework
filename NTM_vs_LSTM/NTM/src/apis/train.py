import torch
import os
import numpy as np
import cv2
import logging
from ..datasets import RandomDataset, RandomDataloader
from ..models import NTM, LossModel, EncapsulateNTM

def init_logger(args):
    logger = logging.Logger(__name__)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path, 'w')
        logger.addHandler(file_handler)
    return logger

def evaluate(outputs, inputs):
    outputs[outputs>0.5] = 1
    outputs[outputs<=0.5] = 0
    correct = (outputs == inputs).sum().item()
    return correct

def clip_grad(net):
    parameters = filter(lambda p: p.grad is not None, net.parameters())
    for p in parameters:
        p.grad.data.clamp_(-10, 10)

def train_old(args):
    logger = init_logger(args)
    train_dataset = RandomDataset()
    model = NTM(args.base_model_type, 1, args.input_dim + 1,  args.input_dim, args.controller_dim,
                args.unit_size, args.memory_len, args.controller_num_layers).to(args.device)
    if args.load_from:
        checkpoint = torch.load(args.load_from)
        for n,p in model.state_dict().items():
            p.copy_(checkpoint[n])
    loss_model = LossModel().to(args.device)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.95))
    optimizer = torch.optim.RMSprop(model.parameters(), args.lr, momentum=0.9, alpha=0.95)
    for epoch in range(args.total_epoch):
        total_loss = 0
        total_acc = 0
        display_loss = 0
        display_acc = 0
        for i in range(len(train_dataset)):
            optimizer.zero_grad()
            batch_loss = 0
            batch_acc = 0
            for j, (inputs, labels) in enumerate(train_dataset):
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                if j >= args.batch_size:
                    break
                outputs = model(inputs)
                loss = loss_model(outputs, labels)
                loss.backward()
                clip_grad(model)
                batch_loss += loss.item()
                correct = evaluate(outputs, labels)
                acc = correct / len(outputs.reshape(-1))
                batch_acc += acc
            batch_loss /= args.batch_size
            batch_acc /= args.batch_size
            display_loss += batch_loss
            display_acc += batch_acc
            optimizer.step()
            total_loss += batch_loss
            total_acc += batch_acc
            if i % args.display_iters == 0 and i > 0:
                logger.info('Epoch:{}, [{}]/[{}], loss={}, acc={}'.format(epoch, i, len(train_dataset),
                                                                    display_loss / args.display_iters,
                                                                    display_acc / args.display_iters))
                display_loss = 0.
                display_acc = 0.
            if i % args.save_iters == 0 and i > 0:
                logger.info('save checkpoint {}/epoch_{}_iter_{}.pth'.format(args.checkpoint_dir, epoch, i))
                torch.save(model.state_dict(), os.path.join(os.path.join(args.checkpoint_dir, 'epoch_{}_iter_{}.pth'.format(epoch, i))))
        logger.info('Train, Epoch{}, loss={}, acc={}'.format(epoch, total_loss/len(train_dataset), total_acc/len(train_dataset)))

def get_outputs(model, inputs):
    in_seq_len = inputs.shape[0]
    out_seq_len = in_seq_len - 1
    model.init_sequence(inputs.size(1), inputs.device)
    for i in range(in_seq_len):
        model(inputs[i])
    outputs = torch.zeros((inputs.shape[0] - 1, inputs.shape[1], inputs.shape[2]-1)).to(inputs.device)
    for i in range(out_seq_len):
        outputs[i], _ = model()
    return outputs


def train(args):
    logger = init_logger(args)
    train_dataloader = RandomDataloader(args.total_iters, args.batch_size, args.input_dim, 1, 20)
    model = EncapsulateNTM(args.base_model_type, args.batch_size, args.input_dim + 1,  args.input_dim, args.controller_dim,
                args.unit_size, args.memory_len, args.controller_num_layers).to(args.device)
    print(model)
    if args.load_from:
        checkpoint = torch.load(args.load_from)
        for n,p in model.state_dict().items():
            p.copy_(checkpoint[n])
    loss_model = LossModel().to(args.device)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.95))
    optimizer = torch.optim.RMSprop(model.parameters(), args.lr, momentum=0.9, alpha=0.95)
    display_loss = 0.
    display_acc = 0.
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        optimizer.zero_grad()
        outputs = get_outputs(model, inputs)
        # outputs = model(inputs)
        loss = loss_model(outputs, labels)
        loss.backward()
        clip_grad(model)
        optimizer.step()
        correct = evaluate(outputs, labels)
        acc = correct / len(outputs.reshape(-1))
        display_loss += loss.item()
        display_acc += acc
        if i % args.display_iters == 0 and i > 0:
            logger.info('Iter [{}]/[{}], loss={}, acc={}'.format(i, args.total_iters,
                                                                display_loss / args.display_iters,
                                                                display_acc / args.display_iters))
            display_loss = 0.
            display_acc = 0.

        if i % args.save_iters == 0 and i > 0:
            logger.info('save checkpoint {}/iter_{}.pth'.format(args.checkpoint_dir, i))
            torch.save(model.state_dict(),
                       os.path.join(os.path.join(args.checkpoint_dir, 'iter_{}.pth'.format(i))))

def test(args, data):
    data = data.to(args.device)
    print('data.shape={}'.format(data.shape))
    model = EncapsulateNTM(args.base_model_type, args.batch_size, args.input_dim + 1,  args.input_dim, args.controller_dim,
                args.unit_size, args.memory_len, args.controller_num_layers).to(args.device)
    checkpoint = torch.load(args.test_pth)
    print('checkpoint.state_dict={}'.format(checkpoint.keys()))
    print('model.state_dict={}'.format(model.state_dict().keys()))
    for n,p in model.state_dict().items():
        p.data.copy_(checkpoint[n])
    model = model.to(args.device)
    model.eval()
    with torch.no_grad():
        outputs = get_outputs(model, data).cpu()
        # outputs = model(data).cpu()
    outputs[outputs>0.5] = 255
    outputs[outputs<=0.5] = 0
    numpy_data = np.array(data[:-1, :, :-1].cpu().reshape(-1, 8)*255).transpose().astype(np.uint8)
    numpy_outputs = np.array(outputs.reshape(-1, 8)).transpose().astype(np.uint8)
    print('numpy_data.shape={}'.format(numpy_data.shape))
    print('numpy_outputs.shape={}'.format(numpy_outputs.shape))
    print(numpy_data)
    print('\n')
    print(numpy_outputs)
    cv2.imwrite('sequnce.jpg', numpy_data)
    cv2.imwrite('output.jpg', numpy_outputs)
    print('acc={}'.format((numpy_data == numpy_outputs).sum() / float(numpy_data.size)))
    numpy_diff = (numpy_data == numpy_outputs).astype(np.uint8)* 255
    cv2.imwrite('diff.jpg', numpy_diff)
    print('numpy_diff={}'.format(255 - numpy_diff))

def test_old(args, data):
    data = data.to(args.device)
    print('data.shape={}'.format(data.shape))
    model = NTM(args.base_model_type, 1, args.input_dim + 2,  args.input_dim, args.controller_dim,
                args.unit_size, args.memory_len, args.controller_num_layers).to(args.device)
    checkpoint = torch.load(args.test_pth)
    print('checkpoint.state_dict={}'.format(checkpoint.keys()))
    print('model.state_dict={}'.format(model.state_dict().keys()))
    for n,p in model.state_dict().items():
        p.copy_(checkpoint[n])
    model = model.to(args.device)
    model.eval()
    with torch.no_grad():
        outputs = model(data).cpu()
    print(outputs)
    outputs[outputs>0.5] = 255
    outputs[outputs<=0.5] = 0
    print(data.shape)
    numpy_data = np.array(data[:, 1:-1, :-2].cpu().reshape(-1, 8)*255).transpose().astype(np.uint8)
    numpy_outputs = np.array(outputs.reshape(-1, 8)).transpose().astype(np.uint8)
    print(numpy_data.shape)
    print(numpy_outputs.shape)
    print(numpy_data)
    print('\n')
    print(numpy_outputs)
    cv2.imwrite('sequnce.jpg', numpy_data)
    cv2.imwrite('output.jpg', numpy_outputs)
    print('acc={}'.format((numpy_data == numpy_outputs).sum() / float(numpy_data.size)))
    numpy_diff = (numpy_data == numpy_outputs).astype(np.uint8)* 255
    cv2.imwrite('diff.jpg', numpy_diff)
    print('numpy_diff={}'.format(255 - numpy_diff))