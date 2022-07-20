import torch
from torch import nn, optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





def get_batches(data, batch_size):
    '''
    批量获取训练集数据，实现采样 ( 70% )
    '''

    return ''


def train(model, data, batch_size, seq_len, epochs, lr=0.01, valid=None):
    '''
    参数说明
    -----------
    model :
    data  :
    batch_size : 一个batch多少个数据
    seq_len : 序列长度（步长）
    epochs : 训练循环次数
    lr : 学习率
    valid : 验证数据
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 损失函数：待定
    criterion = nn.CrossEntropyLoss()
    # 判断是否有valid数据（即是否边训练边验证）
    if valid is not None:
        data = model.onehot_encode(data.reshape(-1, 1))
        valid = model.onehot_encode(valid.reshape(-1, 1))
    else:
        data = model.onehot_encode(data.reshape(-1, 1))

    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        model.train()
        hs = None  # hs 等于 hidden_size,隐藏层结点
        train_ls = 0.0
        val_ls = 0.0
        for x, y in get_batches(data, batch_size):

            optimizer.zero_grad()
            x = torch.tensor(x).float().to(device)

            # 模型训练
            out, hs = model(x, y, teacher_forcing_ratio=0.6)
            hs = ([h.data for h in hs])  # 读取每一个hidden_size的结点

            y = y.reshape(-1, len(model.vocab))
            y = model.onehot_decode(y)
            y = model.label_encode(y.squeeze())
            y = torch.from_numpy(y).long().to(device)

            # 计算损失函数
            loss = criterion(out, y.squeeze())
            loss.backward()
            optimizer.step()
            train_ls += loss.item()

        if valid is not None:
            model.eval()
            hs = None
            with torch.no_grad():
                for x, y in get_batches(valid, batch_size, seq_len):
                    x = torch.tensor(x).float().to(device)
                    out, hs = model(x, hs)  # 预测输出
                    hs = ([h.data for h in hs])

                    y = y.reshape(-1, len(model.vocab))
                    y = model.onehot_decode(y)
                    y = model.label_encode(y.squeeze())
                    y = torch.from_numpy(y).long().to(device)

                    loss = criterion(out, y.squeeze())
                    val_ls += loss.item()

                val_loss.append(np.mean(val_ls))  # 求出每一轮的损失均值，并累计
            train_loss.append(np.mean(train_ls))  # 求出每一轮的损失均值，并累计

        print(f'--------------Epochs{epochs} | {epoch}---------------')
        print(f'Train Loss : {train_loss[-1]}')  # 这里-1为最后添加进去的loss值，即本轮batch的loss
        if val_loss:
            print(f'Val Loss : {val_loss[-1]}')

    #
    # plt.plot(train_loss, label='Train Loss')
    # plt.plot(val_loss, label='Val Loss')
    # plt.title('Loss vs Epochs')
    # plt.legend()
    # plt.show()