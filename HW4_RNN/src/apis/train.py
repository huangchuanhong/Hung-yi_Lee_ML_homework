import os
import torch
from torch.nn.utils import clip_grad
import numpy as np
from torch.utils.data.dataloader import DataLoader
from gensim.models import word2vec
from ..datasets import SentenceDataset, SentenceTestDataset, get_train_label, get_test_data, get_train_nolabel, get_embedding_matrix,\
    train_word2vec
from ..models import LSTM_Net

# def val(dataloader, net):
#     correct = 0
#     total = 0
#     for i, batch in enumerate(dataloader):
#         logits = net(batch[0])
#         predict = logits > 0.5
#         correct += (predict == batch[1]).sum()
#         total += len(batch[1])
#     return correct / total

def evaluate(predict, label):
    predict[predict>0.5] = 1.
    predict[predict<=0.5] = 0.
    correct = torch.eq(predict, label).sum()
    return correct.item()

def test(net, dataloader, device):
    net.eval()
    logits = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            logits.append(net(data.to(device)).squeeze())
    logits = torch.cat(logits).cpu().numpy()
    return logits

def load_checkpoint(net, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    own_dict = net.state_dict()
    for n, p in state_dict.items():
        assert(n in own_dict)
        own_dict[n].copy_(p)

def filter_high_score_idxes(logits, pos_score_thr, neg_score_thr, data_list):
    print('type(logits)={}'.format(type(logits)))
    pos_filtered_idxes = np.where(logits > pos_score_thr)
    neg_filtered_idxes = np.where(logits < neg_score_thr)
    pos_filtered_data = list(np.array(data_list)[pos_filtered_idxes])
    pos_filtered_label = list(np.ones(len(pos_filtered_data), dtype=np.int32))
    neg_filtered_data = list(np.array(data_list)[neg_filtered_idxes])
    neg_filtered_label = list(np.zeros(len(neg_filtered_data), dtype=np.int32))
    print('len(neg_filtered_data)={}'.format(len(neg_filtered_data)))
    print('len(pos_filtered_data)={}'.format(len(pos_filtered_data)))
    print('pos_filtered_data[0]={}'.format(pos_filtered_data[0]))
    print('neg_filtered_data[0]={}'.format(neg_filtered_data[0]))
    filtered_data = pos_filtered_data + neg_filtered_data
    filtered_label = pos_filtered_label + neg_filtered_label
    return filtered_data, filtered_label

def clip_grads(params):
    clip_grad.clip_grad_norm_(
        filter(lambda p: p.requires_grad, params),
        max_norm=15,
        norm_type=2
    )

def train(args):
    train_list, label_list = get_train_label(args.train_label_path)
    test_list = get_test_data(args.test_data_path)
    if args.wv_model_path:
        wv_model = word2vec.Word2Vec.load(args.wv_model_path)
    else:
        wv_model = train_word2vec(train_list + test_list)
    cut_point = int(len(train_list) * args.train_val_ratio)
    val_list = train_list[cut_point:]
    val_label_list = label_list[cut_point:]
    train_list = train_list[:cut_point]
    train_label_list = label_list[:cut_point]

    train_dataset = SentenceDataset(train_list, train_label_list, wv_model, args.sentence_len)
    val_dataset = SentenceDataset(val_list, val_label_list, wv_model, args.sentence_len)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    embedding = get_embedding_matrix(wv_model)

    net = LSTM_Net(embedding, args.embedding_dim, args.hidden_dim, args.lstm_num_layers, dropout=args.dropout, \
                   fix_embedding=args.fix_embedding)
    net.to(args.device)
    criterion = torch.nn.BCELoss().to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), args.lr)
    best_correct = 0
    for epoch in range(args.epoch):
        train_correct = 0.
        total_loss = 0.
        for i, batch in enumerate(train_dataloader):
            data, label = batch
            data = data.to(args.device)
            label = label.to(args.device)
            optimizer.zero_grad()
            logits = net(data).squeeze()
            loss = criterion(logits, label)
            total_loss += loss.item()
            loss.backward()
            clip_grads(net.parameters())
            optimizer.step()
            correct = evaluate(logits, label)
            train_correct += correct
            if i % args.display_iters == 0 and i > 0:
                print('epoch {}: [{}/{}] loss={:.3f} acc={:3f}'.format(epoch, i, len(train_dataloader), loss.item(), correct/len(data)))
        print('\nEpoch{} Train | loss={:.5f} acc={:.5f}'.format(epoch, total_loss/len(train_dataloader), train_correct/len(train_dataset)))

        net.eval()
        with torch.no_grad():
            val_correct = 0.
            val_loss = 0.
            for i, (data, label) in enumerate(val_dataloader):
                data = data.to(args.device)
                label = label.to(args.device)
                logits = net(data).squeeze()
                loss = criterion(logits, label)
                val_loss += loss.item()
                val_correct += evaluate(logits, label)
        print('\n Val | loss={:.5f} acc={:.5f}'.format(val_loss/len(val_dataloader), val_correct/len(val_dataset)))

        if val_correct > best_correct:
            best_correct = val_correct
            torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, 'ckpt.pth'))
            print('Saving checkpoint with acc:{:.5f}'.format(val_correct/len(val_dataset)))
        net.train()

def self_learning_train(args):
    torch.manual_seed(0)
    train_list, label_list = get_train_label(args.train_label_path)
    test_list = get_test_data(args.test_data_path)
    train_nolabel_list = get_train_nolabel(args.train_nolabel_path)
    if args.wv_model_path:
        wv_model = word2vec.Word2Vec.load(args.wv_model_path)
    else:
        wv_model = train_word2vec(train_list + test_list + train_nolabel_list)
    cut_point = int(len(train_list) * args.train_val_ratio)
    val_list = train_list[cut_point:]
    val_label_list = label_list[cut_point:]
    train_list = train_list[:cut_point]
    train_label_list = label_list[:cut_point]

    embedding_matrix = get_embedding_matrix(wv_model)
    net = LSTM_Net(embedding_matrix, args.embedding_dim, args.hidden_dim, args.lstm_num_layers, dropout=args.dropout, \
                   fix_embedding=args.fix_embedding)
    net.to(args.device)
    init_param = {}
    for n, p in net.state_dict().items():
        init_param[n] = torch.zeros_like(p)
        init_param[n].copy_(p)
    criterion = torch.nn.BCELoss().to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), args.lr)

    train_nolabel_dataset = SentenceTestDataset(train_nolabel_list, wv_model, args.sentence_len)
    train_nolabel_dataloader = DataLoader(train_nolabel_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    cur_train_list = train_list.copy()
    cur_train_label_list = train_label_list.copy()
    # best_correct = 0
    for sl_iter in range(args.self_learning_iters):
        best_correct = 0
        train_dataset = SentenceDataset(cur_train_list, cur_train_label_list, wv_model, args.sentence_len)
        val_dataset = SentenceDataset(val_list, val_label_list, wv_model, args.sentence_len)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

        for epoch in range(args.epoch):
            train_correct = 0.
            total_loss = 0.
            for i, batch in enumerate(train_dataloader):
                data, label = batch
                data = data.to(args.device)
                label = label.to(args.device)
                optimizer.zero_grad()
                logits = net(data).squeeze()
                loss = criterion(logits, label)
                total_loss += loss.item()
                loss.backward()
                clip_grads(net.parameters())
                optimizer.step()
                correct = evaluate(logits, label)
                train_correct += correct
                if i % args.display_iters == 0 and i > 0:
                    print('epoch {}: [{}/{}] loss={:.3f} acc={:3f}'.format(epoch, i, len(train_dataloader), loss.item(), correct/len(data)))
            print('\nEpoch{} Train | loss={:.5f} acc={:.5f}'.format(epoch, total_loss/len(train_dataloader), train_correct/len(train_dataset)))

            net.eval()
            with torch.no_grad():
                val_correct = 0.
                val_loss = 0.
                for i, (data, label) in enumerate(val_dataloader):
                    data = data.to(args.device)
                    label = label.to(args.device)
                    logits = net(data).squeeze()
                    loss = criterion(logits, label)
                    val_loss += loss.item()
                    val_correct += evaluate(logits, label)
            print('\n Val | loss={:.5f} acc={:.5f}'.format(val_loss/len(val_dataloader), val_correct/len(val_dataset)))

            if val_correct > best_correct:
                best_correct = val_correct
                torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, 'ckpt_{}.pth'.format(sl_iter)))
                print('Saving checkpoint with acc:{:.5f}'.format(val_correct/len(val_dataset)))
            net.train()
        # draw data with large scores from nolabel dataset
        load_checkpoint(net, os.path.join(args.checkpoint_dir, 'ckpt_{}.pth'.format(sl_iter)))
        net.to(args.device)
        net.eval()
        with torch.no_grad():
            logits = test(net, train_nolabel_dataloader, args.device)
            filtered_train_list, filtered_label_list = filter_high_score_idxes(logits,
                                                                               args.nolabel_pos_score_thr,
                                                                               args.nolabel_neg_score_thr,
                                                                               train_nolabel_list)
            cur_train_list = train_list + filtered_train_list
            cur_train_label_list = train_label_list + filtered_label_list
            # for data in filtered_train_list:
            #     train_nolabel_list.remove(data)
        # for n, p in net.state_dict().items():
        #     p.copy_(init_param[n])
        # net.train()
        # net = LSTM_Net(embedding_matrix, args.embedding_dim, args.hidden_dim + 10 * (sl_iter + 1), args.lstm_num_layers, dropout=args.dropout, \
        #            fix_embedding=args.fix_embedding).to(args.device)
        net = LSTM_Net(embedding_matrix, args.embedding_dim, args.hidden_dim, args.lstm_num_layers, dropout=args.dropout, \
                       fix_embedding=args.fix_embedding).to(args.device)
        optimizer = torch.optim.Adam(net.parameters(), args.lr)
        net.train()


