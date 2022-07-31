from subseq2seq_attention import *
import torch
import pandas as pd
import re
import math
from train_model import *
from data_processing import *


def train(model, n_epochs, data, optimizer, criterion):
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        cnt = 0
        print('********** epoch: {} **********'.format(epoch))
        for x, y, trg_len, timestamp in get_batches(data, 8, 0.7):
            optimizer.zero_grad()
            x = x.transpose(0, 1)
            y = y.transpose(0, 1)
            output = model(src=x, trg=y, timestamp=timestamp)
            y = y.transpose(0, 1)
            output = output.transpose(0, 1)

            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            epoch_loss += loss.item()
            print('{} batch loss: {}, '.format(cnt, loss.item()), end=' ')
            cnt += 1
        epoch_loss = epoch_loss / cnt
        print('----------------epoch {} loss: {} ---------------'.format(epoch, epoch_loss))
        # torch.save(model.state_dict(), './saved_models/1_model_epoch{}.pth'.format(epoch))
        torch.save(model, './saved_models/2_model_epoch{}.pth'.format(epoch))


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


mlp = MLP(input_size=4, hidden_size=16, output_size=2)
encoder = Encoder(input_dim=2, enc_hid_dim=4, num_layers=2, dropout=0.5, dec_hid_dim=4, bidirectional=True)
attn = Attention(enc_hid_dim=4, dec_hid_dim=4)
decoder = Decoder(output_dim=2, dec_input_dim=2, enc_hid_dim=4, dec_hid_dim=4, num_layers=1, dropout=0.5,
                  attention=attn, mlp=mlp, bidirectional=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SubSeq2Seq(encoder, decoder, device)
train_set, val_set, test_set = split_data_set()
n_epochs = 5
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


model = model.to(device)
criterion = criterion.to(device)

train(model, n_epochs, train_set, optimizer, criterion)

# val_model = torch.load()

# s = torch.tensor([[1, 2, 3, 4]], dtype=torch.float)
# dec_in = torch.tensor([[5, 10]], dtype=torch.float)
# enc_out = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 3, 4, 5, 6, 7, 8]]], dtype=torch.float)
# pre = torch.tensor([[6, 9]], dtype=torch.float)
# suc = torch.tensor([[5, 14]], dtype=torch.float)





# model.train()
# epoch_loss = 0
# cnt = 0
#
# for epoch in range(n_epochs):
#
#
# for x, y, trg_len, timestamp in get_batches(data, 8, 0.7):
#     optimizer.zero_grad()
#     x = x.transpose(0, 1)
#     y = y.transpose(0, 1)
#     output = model(src=x, trg=y, timestamp=timestamp)
#     y = y.transpose(0, 1)
#     output = output.transpose(0, 1)
#     print(output.shape)
#     print(y.shape)
#     loss = criterion(output,  y)
#     loss.backward()
#     epoch_loss += loss.item()
#     print('--------------- {} -----------------'.format(cnt))
#     print('loss: {}'.format(loss.item()))
#     cnt += 1
#
# torch.save(model.state_dict(), './saved_models/model_0.pth')



# enc_output = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8]], [[9, 10, 11, 12, 13, 14, 15, 16]]])

# res = decoder(dec_in, s, enc_out, pre, suc)
#
# print(res.shape)



# x, y, trg_len, timestamp = get_batches(data, 3, 0.7)
#
#
# print('-----------')
# # print(x)
# # print('-----------')
# # print(y)
# # print('-----------')
# # print(trg_len)
# # print('-----------')
# # print(timestamp)
#
# res = model(src=x, trg=y, timestamp=timestamp)
# print('result: ')
# res = res.transpose(0, 1)
# print(res.shape)
# print(res)
#
# y = y.transpose(0, 1)
#
# loss = criterion(res, y)
# print(loss)


# res = model()
# print(res)

# a = [[1, 2, 3, 4], [5, 6, 7, 8]]
# a = torch.tensor(a, dtype=torch.float)
# a = a.unsqueeze(1)
# print(a.shape)
#
# # _, y = model(a)
# print(y)
# print(y.shape)



#

# print(x)
# print('-----------')
# print(y)
# print('----------------')
# print(len)


