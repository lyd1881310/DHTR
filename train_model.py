import torch
from torch import nn, optim
import torch.nn.functional as F

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from DHTR import *
from data_processing import *


def get_batches(data, batch_size, sample_ratio=0.7):
    '''
    批量获取训练集数据，实现采样 ( 70% )
    data: list类型
    '''
    data_size = len(data) - batch_size

    for i in range(0, data_size, batch_size):
        min_len = 999999
        x = []
        ground_truth = []
        for j in range(0, batch_size):  # 获取该批次中序列的最小长度
            min_len = min(min_len, len(data[i + j]))
        for j in range(0, batch_size):
            ground_truth.append(data[i + j][0: min_len])
        sample_num = int(min_len * sample_ratio)
        sample = []
        sample.append(0)
        t = random.sample(range(1, min_len - 1), sample_num)
        for u in t:
            sample.append(u)
        sample.append(min_len - 1)

        sample.sort()
        # sample = np.random.randint(1, min_len - 1, sample_num)
        for j in range(0, batch_size):
            tra = []
            for k in sample:
                tra.append(ground_truth[j][k])
            x.append(tra)
        # return x, ground_truth, min_len
        # print(min_len)
        # print(sample)
        yield torch.tensor(x, dtype=torch.float), torch.tensor(ground_truth, dtype=torch.float), \
               min_len, sample


# def train(model, data, batch_size, seq_len, epochs, lr=0.01, valid=None):
#     '''
#     参数说明
#     -----------
#     model :
#     data  :
#     batch_size : 一个batch多少个数据
#     seq_len : 序列长度（步长）
#     epochs : 训练循环次数
#     lr : 学习率
#     valid : 验证数据
#     '''
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#
#     # 损失函数：待定
#     criterion = nn.CrossEntropyLoss()
#     # 判断是否有valid数据（即是否边训练边验证）
#     if valid is not None:
#         data = model.onehot_encode(data.reshape(-1, 1))
#         valid = model.onehot_encode(valid.reshape(-1, 1))
#     else:
#         data = model.onehot_encode(data.reshape(-1, 1))
#
#     train_loss = []
#     val_loss = []
#     for epoch in range(epochs):
#         model.train()
#         hs = None  # hs 等于 hidden_size,隐藏层结点
#         train_ls = 0.0
#         val_ls = 0.0
#         for x, y in get_batches(data, batch_size):
#
#             optimizer.zero_grad()
#             x = torch.tensor(x).float().to(device)
#
#             # 模型训练
#             out, hs = model(x, y, teacher_forcing_ratio=0.6)
#             hs = ([h.data for h in hs])  # 读取每一个hidden_size的结点
#
#             y = y.reshape(-1, len(model.vocab))
#             y = model.onehot_decode(y)
#             y = model.label_encode(y.squeeze())
#             y = torch.from_numpy(y).long().to(device)
#
#             # 计算损失函数
#             loss = criterion(out, y.squeeze())
#             loss.backward()
#             optimizer.step()
#             train_ls += loss.item()
#
#         if valid is not None:
#             model.eval()
#             hs = None
#             with torch.no_grad():
#                 for x, y in get_batches(valid, batch_size, seq_len):
#                     x = torch.tensor(x).float().to(device)
#                     out, hs = model(x, hs)  # 预测输出
#                     hs = ([h.data for h in hs])
#
#                     y = y.reshape(-1, len(model.vocab))
#                     y = model.onehot_decode(y)
#                     y = model.label_encode(y.squeeze())
#                     y = torch.from_numpy(y).long().to(device)
#
#                     loss = criterion(out, y.squeeze())
#                     val_ls += loss.item()
#
#                 val_loss.append(np.mean(val_ls))  # 求出每一轮的损失均值，并累计
#             train_loss.append(np.mean(train_ls))  # 求出每一轮的损失均值，并累计
#
#         print(f'--------------Epochs{epochs} | {epoch}---------------')
#         print(f'Train Loss : {train_loss[-1]}')  # 这里-1为最后添加进去的loss值，即本轮batch的loss
#         if val_loss:
#             print(f'Val Loss : {val_loss[-1]}')

    #
    # plt.plot(train_loss, label='Train Loss')
    # plt.plot(val_loss, label='Val Loss')
    # plt.title('Loss vs Epochs')
    # plt.legend()
    # plt.show()


def train(model, n_epochs, data, optimizer, criterion, log, device):
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        cnt = 0
        print('********** epoch: {} **********'.format(epoch))
        log.write('********** epoch: {} **********\n'.format(epoch))
        for x, y, trg_len, timestamp in get_batches(data, 16, 0.7):
            optimizer.zero_grad()
            # x = x.transpose(0, 1)
            # y = y.transpose(0, 1)
            output, cell_res = model(src=x, trg=y, timestamp=timestamp)
            # y = y.transpose(0, 1)
            # output = output.transpose(0, 1)

            # print('loss')
            # print(y.shape)
            # print(cell_res.shape)
            cell_res = cell_res.to(device)
            y = y.to(device)
            loss = criterion(cell_res, y)
            # print(loss)
            # print(type(loss))
            # print(loss.shape)
            loss = loss.requires_grad_(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            epoch_loss += loss.item()
            if cnt % 100 == 0:
                print('{} batch loss: {}'.format(cnt, loss.item()))
                log.write('{} batch loss: {}, \n'.format(cnt, loss.item()))
            cnt += 1
        epoch_loss = epoch_loss / cnt
        print('----------------epoch {} loss: {} ---------------'.format(epoch, epoch_loss))
        log.write('----------------epoch {} loss: {} ---------------\n'.format(epoch, epoch_loss))
        # torch.save(model.state_dict(), './saved_models/1_model_epoch{}.pth'.format(epoch))
        torch.save(model, './saved_models/4_model_epoch{}.pth'.format(epoch))


def eval(model, data, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        batch_num = 0
        for x, y, trg_len, timestamp in get_batches(data, 8, 0.7):
            x = x.transpose(0, 1)
            y = y.transpose(0, 1)
            output = model(src=x, trg=y, timestamp=timestamp)
            y = y.transpose(0, 1)
            output = output.transpose(0, 1)
            loss = criterion(output, y)

            print('eval {} batch loss: {}'.format(batch_num, loss.item))
            batch_num += 1
            epoch_loss += loss.item()

        print('eval loss: {}'.format(epoch_loss / batch_num))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(device=device)
    attn = Attention(device=device)
    mlp = MLP()
    decoder = Decoder(device=device, attention=attn, mlp=mlp)
    model = DHTR(encoder=encoder, decoder=decoder, device=device)

    # model = torch.load('./saved_models/3_2_model_epoch6.pth')

    train_set, val_set, test_set = split_data_set()
    n_epochs = 10
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    fout = open('log.txt', 'w')

    model = model.to(device)
    criterion = criterion.to(device)

    cnt = 0
    for param in model.parameters():
        cnt += 1
        param.requires_grad = True

    print(cnt)

    train(model, n_epochs, train_set, optimizer, criterion, fout, device)