import torch
import os
import numpy as np
import cv2
from ..datasets import RandomDataset
from ..models import LossModel, LSTM_Net

def evaluate(outputs, inputs):
    outputs[outputs>0.5] = 1
    outputs[outputs<=0.5] = 0
    correct = (outputs == inputs).sum().item()
    return correct

def train(args):
    train_dataset = RandomDataset()
    model = LSTM_Net(args.input_dim, args.hidden_dim, args.num_layers).to(args.device)
    if args.load_from:
        checkpoint = torch.load(args.load_from)
        for n,p in model.state_dict().items():
            p.copy_(checkpoint[n])
    loss_model = LossModel().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    for epoch in range(args.total_epoch):
        total_loss = 0
        total_acc = 0
        for i in range(len(train_dataset)):
            optimizer.zero_grad()
            batch_loss = 0
            batch_acc = 0
            for j, inputs in enumerate(train_dataset):
                inputs = inputs.to(args.device)
                if j > args.batch_size:
                    break
                outputs = model(inputs)
                loss = loss_model(outputs, inputs[:,:-1:])
                loss.backward()
                batch_loss += loss.item()
                correct = evaluate(outputs, inputs[:, :-1, :])
                acc = correct / len(outputs.reshape(-1))
                batch_acc += acc
            batch_loss /= args.batch_size
            batch_acc /= args.batch_size
            optimizer.step()
            total_loss += batch_loss
            total_acc += batch_acc
            if i % args.display_iters == 0 and i > 0:
                print('Epoch:{}, [{}]/[{}], loss={}, acc={}'.format(epoch, i, len(train_dataset), batch_loss, batch_acc))
            if i % args.save_iters == 0 and i > 0:
                torch.save(model.state_dict(), os.path.join(os.path.join(args.checkpoint_dir, 'epoch_{}.pth'.format(i))))
        print('Train, Epoch{}, loss={}, acc={}'.format(epoch, total_loss/len(train_dataset), total_acc/len(train_dataset)))


        # for i, inputs in enumerate(train_dataset):
        #     if i > 100000:
        #         break
        #     optimizer.zero_grad()
        #     outputs = model(inputs)
        #     loss = loss_model(outputs, inputs[:,:-1,:])
        #     loss.backward()
        #     optimizer.step()
        #     total_loss += loss.item()
        #     correct = evaluate(outputs, inputs[:, :-1, :])
        #     acc = correct / len(outputs.reshape(-1))
        #     total_acc += acc
        #     if i % args.display_iters == 0 and i > 0:
        #         print('Epoch:{}, [{}]/[{}], loss={}, acc={}'.format(epoch, i, len(train_dataset), loss.item(), acc))
        # print('Train, Epoch{}, loss = {}, acc={}'.format(epoch, total_loss/len(train_dataset), total_acc/len(train_dataset)))

def test(args, data):
    data = data.to(args.device)
    model = LSTM_Net(args.input_dim, args.hidden_dim, args.num_layers)
    checkpoint = torch.load(args.test_pth)
    for n,p in model.state_dict().items():
        p.copy_(checkpoint[n])
    model = model.to(args.device)
    model.eval()
    with torch.no_grad():
        outputs = model(data).cpu()
    outputs[outputs>0.5] = 255
    outputs[outputs<=0.5] = 0
    print(data)
    numpy_data = np.array(data.cpu().reshape(-1, 8)*255).transpose().astype(np.uint8)
    numpy_outputs = np.array(outputs.reshape(-1, 8)).transpose().astype(np.uint8)
    print(numpy_data[:, :-1].shape)
    print(numpy_outputs.shape)
    print(numpy_data[:, :-1])
    print(numpy_outputs)
    cv2.imwrite('sequnce.jpg', numpy_data[:, :-1])
    cv2.imwrite('output.jpg', numpy_outputs)


