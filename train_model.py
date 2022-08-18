import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

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


def train(model, n_epochs, data, optimizer, criterion, log, device, batch_size):
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        cnt = 0
        print('********** epoch: {} **********'.format(epoch))
        log.write('********** epoch: {} **********\n'.format(epoch))
        for x, y, trg_len, timestamp in get_batches(data, batch_size, 0.7):
            optimizer.zero_grad()
            output, cell_res = model(src=x, trg=y, timestamp=timestamp)
            cell_res = cell_res.to(device)
            y = y.to(device)
            loss = criterion(cell_res, y)

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


if __name__ == '__main__':
    n_epochs = 10
    batch_size = 16
    learning_rate = 0.05

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(device=device)
    attn = Attention(device=device)
    mlp = MLP()
    decoder = Decoder(device=device, attention=attn, mlp=mlp)
    model = DHTR(encoder=encoder, decoder=decoder, device=device)

    train_set, val_set, test_set = split_data_set()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    log = open('log.txt', 'w')

    model = model.to(device)
    criterion = criterion.to(device)

    cnt = 0
    for param in model.parameters():
        cnt += 1
        param.requires_grad = True

    train(model, n_epochs, train_set, optimizer, criterion, log, device, batch_size)