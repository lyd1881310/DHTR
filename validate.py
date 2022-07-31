from subseq2seq_attention import *
import torch
import pandas as pd
import re
import math
from train_model import *
from data_processing import *

mlp = MLP(input_size=4, hidden_size=16, output_size=2)
encoder = Encoder(input_dim=2, enc_hid_dim=4, num_layers=2, dropout=0.5, dec_hid_dim=4, bidirectional=True)
attn = Attention(enc_hid_dim=4, dec_hid_dim=4)
decoder = Decoder(output_dim=2, dec_input_dim=2, enc_hid_dim=4, dec_hid_dim=4, num_layers=1, dropout=0.5,
                  attention=attn, mlp=mlp, bidirectional=False)
# model = SubSeq2Seq(encoder, decoder, 'cuda')
# model.load_state_dict(torch.load('./saved_models/2_model_epoch4.pth'))
model = torch.load('./saved_models/2_model_epoch4.pth')

# model = torch.load('./saved_models/1_model_epoch9.pth')
print(model)

data_frame = pd.read_csv('./data/data_set_4.csv')
data = []
thresh = 2
cnt = 0
for index, row in data_frame.iterrows():
    data.append(polyline_to_list(row['traj']))
    cnt += 1
    if cnt >= thresh:
        break

# input = torch.tensor(data[0], dtype=torch.float)
for x, y, trg_len, timestamp in get_batches(data, 1, 0.7):
    print(x)
    print(y)
    x = x.transpose(0, 1)
    y = y.transpose(0, 1)
    res = model(x, y, timestamp)
    print('---------')
    print(res)
    loss_func = nn.CrossEntropyLoss()
    y = y.transpose(0, 1)
    loss = loss_func(res, y)
    print(loss)

    # a = torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.float)
    # b = torch.tensor([[[2, 2], [3, 5], [4, 6]]], dtype=torch.float)
    # c = torch.tensor([[[10, 1], [7, 2], [12, 15]]], dtype=torch.float)
    #
    # loss_1 = loss_func(b, a)
    # loss_2 = loss_func(c, a)
    # print(loss_1)
    # print(loss_2)

