import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from utils import *


# Decoder 的多层感知机
class MLP(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, output_size=64):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.relu(x1)
        y = self.linear2(x2)
        return y


class Encoder(nn.Module):
    def __init__(self, device, input_dim=2, enc_hid_dim=256, num_layers=2, dropout=0.5, dec_hid_dim=512, bidirectional=True):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=enc_hid_dim,
                           num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src):
        '''
        src = [batch_size, src_len, input_dim]
        '''

        # enc_output = [batch_size, src_len, hid_dim * num_directions]
        # enc_hidden = [n_layers * num_directions, src_len, hid_dim]
        enc_output, (enc_hidden, c) = self.rnn(src)  # if h_0 is not give, it will be set 0 acquiescently
        # print(enc_output.shape)
        # print(enc_hidden.shape)

        # 双向 LSTM: 合并最后时刻的隐藏层输出
        # s = [batch_size, dec_hid_dim]
        s = enc_output[:, -1, :].squeeze(1)
        return enc_output, s
        # return enc_hidden


# 计算权重 a1, a2, a3, ...
class Attention(nn.Module):
    def __init__(self, device):
        super().__init__()
        # self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.device = device

        self.WH = nn.Linear(512, 256)
        self.WS = nn.Linear(512, 256)
        self.WP = nn.Linear(512, 256)
        self.WQ = nn.Linear(512, 256)

        self.spatial_dist_embedding = nn.Embedding(1024, 512)
        self.temporal_dist_embedding = nn.Embedding(1024, 512)

        # self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        self.v = nn.Linear(256, 1)

    def forward(self, h, s, dec_input, trg_time_step, enc_src, src_timestamp):
        # print('-------------- Attention ----------------')
        # h = [batch_size, h_dim]
        # src_timestamp = [batch_size, src_len]
        # enc_src = [batch_size, src_len, enc_input_dim]
        # trg_time_step = [batch_size]
        # dec_input = [batch_size, dec_input_dim]
        # s --- enc_output  s = [batch_size, src_len, enc_output_dim]

        batch_size = s.shape[0]
        src_len = s.shape[1]
        trg_time_step = torch.tensor([trg_time_step], dtype=torch.float).to(self.device)
        src_timestamp = torch.tensor(src_timestamp, dtype=torch.float).repeat(batch_size, 1).to(self.device)
        trg_time_step = trg_time_step.repeat(batch_size, 1, src_len).squeeze(1).to(self.device)
        # p = [batch_size, src_len]

        # print('trg_time_step: {}'.format(trg_time_step.shape))
        # print('src_timestamp: {}'.format(src_timestamp.shape))
        p = torch.abs(trg_time_step - src_timestamp).squeeze(0)
        # p = torch.LongTensor(p)
        p = p.long().to(self.device)
        p = self.temporal_dist_embedding(p)
        # print('p: {}'.format(p.shape))

        # dec_input = [batch_size, src_len, dec_input_dim]
        # print('dec_input_0: {}'.format(dec_input.shape))
        dec_input = dec_input.unsqueeze(1).repeat(1, src_len, 1)

        # print('dec_input: {}'.format(dec_input.shape))
        # print('enc_src: {}'.format(enc_src.shape))
        t_0 = (dec_input - enc_src) * (dec_input - enc_src)
        # q = [batch_size, src_len]
        q = torch.sqrt(t_0.sum(axis=2, keepdims=False))
        q = q.long().to(self.device)
        q = self.spatial_dist_embedding(q)
        # print('q: {}'.format(q.shape))
        # h = [batch_size, src_len, h_dim]
        # print(h.shape)
        h = h.unsqueeze(1).repeat(1, src_len, 1)

        h = h.to(self.device)
        s = s.to(self.device)
        wh = self.WH(h)
        ws = self.WS(s)
        wp = self.WP(p)
        wq = self.WQ(q)

        v_in = torch.tanh(wh + ws + wp + wq)

        # attention = [batch_size, src_len]
        attention = self.v(v_in).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, attention, mlp, device, output_dim=6351, dec_input_dim=2, enc_hid_dim=512, dec_hid_dim=512, num_layers=1, dropout=0.5, bidirectional=False):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.device = device
        # self.rnn = nn.LSTM(input_size=(enc_hid_dim * 2) + dec_input_dim, hidden_size=dec_hid_dim,
        #                    num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        # self.fc_in = nn.Linear(2 * (dec_input_dim), dec_input_dim)
        self.rnn = nn.LSTM(input_size=512, hidden_size=dec_hid_dim,
                           num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.fc_in = nn.Linear(dec_input_dim + 64, 512)

        # self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + dec_input_dim, output_dim)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        self.mlp = mlp

    def forward(self, dec_input, h, enc_output, precursor, successor, trg_time_step, enc_src, src_timestamp):
        # dec_input = [batch_size, dec_input_dim(2)]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        # h = [batch_size, dec_hid_dim]


        # print('dec_input: {}', dec_input.shape)
        dec_input = dec_input.to(self.device)
        h = h.to(self.device)
        enc_output = enc_output.to(self.device)
        precursor = precursor.to(self.device)
        successor = successor.to(self.device)
        enc_src = enc_src.to(self.device)

        r = self.mlp(torch.cat((precursor, successor), dim=1).to(self.device))
        # print('r: {}', r.shape)
        # dec_input = [batch_size, 1, dec_input_dim(2)]
        rnn_input = self.fc_in(torch.cat((r, dec_input), dim=1).to(self.device)).unsqueeze(1)

        # a = [batch_size, src_len, 1]
        a = torch.tensor(self.attention(h, enc_output, dec_input, trg_time_step, enc_src, src_timestamp)).unsqueeze(2)
        # print('a: {}'.format(a.shape))
        # print('enc_output: {}'.format(enc_output.shape))
        # c = [batch_size, 1, enc_hid_dim * 2]
        c = torch.bmm(a.transpose(1, 2), enc_output)
        # print('rnn_input: {}'.format(rnn_input.shape))
        # print('c: {}'.format(c.shape))

        # rnn_input = torch.cat((rnn_input, c), dim=2)
        rnn_input = rnn_input + c
        # print('input: {}'.format(rnn_input))
        h = h.unsqueeze(0)
        # h = h.unsqueeze(1)
        h_0 = h
        c_0 = h

        # dec_output = [batch_size, 1, dec_out_dim]
        # dec_hidden = [1, batch_size, dec_hid_dim] -- batch_fist 不影响 h, c 的维度
        # print('------------- Decoder -----------')
        rnn_output, (dec_hidden, c_n) = self.rnn(rnn_input, (h_0, c_0))
        # print(dec_hidden.shape)
        rnn_output = rnn_output.squeeze(1)
        dec_hidden = dec_hidden.squeeze(0)

        # dec_output = [batch_size, dec_output_dim(cells_num)]
        dec_output = self.fc_out(rnn_output)

        # print(dec_hidden.shape)
        return F.softmax(dec_output), dec_hidden


class KalmanFilter(nn.Module):
    def __init__(self, z_size, N_size, g_size, H_size):
        super().__init__()
        self.phi = torch.tensor([[0.4, 0.2, 0.1, 0.1],
                                [0.4, 0.2, 0.1, 0.1],
                                [0.4, 0.2, 0.1, 0.1],
                                [0.4, 0.2, 0.1, 0.1]])
        self.psi = torch.tensor([[0.23, 0.3, 0.4, 0.1],
                                 [0.23, 0.3, 0.4, 0.1]])

    def forward(self, z, N, g, H):
        # Predict
        g_predict = torch.matmul(self.phi, g)
        t_1 = torch.matmul(self.phi, H)
        H_predict = torch.matmul(t_1, self.phi.t())   # 计算 H(i|i-1) 考虑协方差矩阵 M 的表示

        # Kalman Gain
        t_2 = torch.matmul(self.psi, H_predict)
        t_2 = torch.matmul(t_2, self.psi.t())
        t_2 = t_2 + N
        K = torch.matmul(H_predict, self.psi.t())
        K = torch.matmul(K, t_2.inverse())

        # Update
        t_3 = torch.matmul(self.psi, g_predict)
        g_update = g_predict + torch.matmul(K, (z - t_3))
        t_4 = torch.matmul(self.psi, H_predict)
        H_update = H_predict - torch.matmul(K, t_4)
        z_update = torch.matmul(self.psi, g_update)
        return g_update, H_update, z_update


class DHTR(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, timestamp, teacher_forcing_ratio=0.5):
        # src = [batch_size, src_len, enc_input_dim]
        # trg = [batch_size, trg_len, dec_output_dim]
        # use teacher forcing
        src = src.to(self.device)
        trg = trg.to(self.device)
        # print('--------- DHTR ----------')
        # print(src.shape)
        # print(trg.shape)

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        dec_output_dim = trg.shape[2]
        pre_index, suc_index = get_pre_suc(timestamp, trg_len)

        # trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        # outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # enc_output = [batch_size, src_len, enc_hid_dim(512)]
        # s_m = [batch_size, enc_hid_dim(512)]
        enc_output, s_m = self.encoder(src)

        # print('$$$$ ')
        # print(type(s))
        # print(s)

        #
        dec_input = src[:, 0, :]
        # outputs = torch.zeros(trg_len, batch_size, dec_output_dim).to(self.device)
        outputs = torch.zeros(batch_size, trg_len, 6351)
        cell_res = torch.zeros(batch_size, trg_len, 2)
        cell_res[:, 0, :] = src[:, 0, :]

        outputs = outputs.to(self.device)
        cell_res = cell_res.to(self.device)

        dec_hidden = s_m
        for t in range(1, trg_len):
            precursor = pre_index[t]
            successor = suc_index[t]

            # dec_output = [batch_size, 6351]
            dec_output, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output, src[:, precursor, :], src[:, successor, :], t, src, timestamp)
            # print(dec_output.shape)

            # Copy 机制
            outputs[:, t, :] = dec_output
            if timestamp.count(t) > 0:
                cell_res[:, t, :] = src[:, timestamp.index(t), :]
            else:
                n = torch.argmax(dec_output, dim=1)
                # print('---------- cell -------------')
                # print(cell_res[:, t, :].shape)
                # print(cell_res[:, t, 0].shape)
                # print(n.shape)
                x = (n // 87).unsqueeze(1)
                y = (n % 87).unsqueeze(1)
                cell_res[:, t, :] = torch.cat((x, y), dim=1)

            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[:, t, :] if teacher_force else cell_res[:, t, :]
            # 是否使用 teacher forcing
            # teacher_force = random.random() < teacher_forcing_ratio
            # top1 = dec_output.argmax(1)
            # dec_input = trg[t] if teacher_force else top1

        return outputs, cell_res


