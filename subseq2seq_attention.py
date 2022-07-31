import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from utils import *


# Decoder 的多层感知机
class MLP(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=512):
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
    def __init__(self, input_dim, enc_hid_dim, num_layers, dropout, dec_hid_dim, bidirectional=True):
        super().__init__()
        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=enc_hid_dim,
                           num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src):
        '''
        src = [src_len, batch_size, input_dim]
        '''
        enc_output, enc_hidden = self.rnn(src)  # if h_0 is not give, it will be set 0 acquiescently
        # enc_hidden 是 tupple 类型，转换为 tensor
        enc_hidden = enc_hidden[0]
        # print(enc_hidden[0])
        # enc_hidden = torch.tensor(enc_hidden)

        # 双向 LSTM: 合并最后时刻的隐藏层输出
        # s = [batch_size, dec_hid_dim]
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))
        # print('-------- Encoder -----------')
        # print(s.shape)

        return enc_output, s
        # return enc_hidden


# 计算权重 a1, a2, a3, ...
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        # batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # 将 Decoder 的隐藏层输入重复 src_len 次
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        # s = s[0]/
        # print('*******')
        # print(s.shape)
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # print('-------------- Attention -------------------')
        # print(s.shape)
        # print(enc_output.shape)

        # energy = [batch_size, src_len, dec_hid_dim]

        attn_input = torch.cat((s, enc_output), dim=2)
        attn_input = torch.tensor(attn_input, dtype=torch.float)
        # print(type(attn_input))
        # print(attn_input.shape)

        attn_output = self.attn(attn_input)
        # print(type(attn_output))
        # print(attn_output.shape)
        energy = torch.tanh(attn_output)

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, dec_input_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention, mlp, bidirectional=False):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.rnn = nn.LSTM(input_size=(enc_hid_dim * 2) + dec_input_dim, hidden_size=dec_hid_dim,
                           num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.fc_in = nn.Linear(2 * (dec_input_dim), dec_input_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + dec_input_dim, output_dim)
        self.mlp = mlp

    def forward(self, dec_input, s, enc_output, precursor, successor):
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        r = self.mlp(torch.cat((precursor, successor), dim=1))
        # print('-------- Decoder ------------')
        # # s = torch.tensor(s, dtype=torch.float)
        # print(s)
        # print(len(s))
        # print(r.shape)
        # print(dec_input.shape)
        dec_input = self.fc_in(torch.cat((r, dec_input), dim=1)).unsqueeze(0)

        a = torch.tensor(self.attention(s, enc_output)).unsqueeze(1)
        # print(a)
        enc_output = enc_output.transpose(0, 1)
        # print(enc_output)
        c = torch.bmm(a, enc_output).transpose(0, 1)

        # print('------------- Decoder -----------')
        # print(dec_input.shape)
        # print(c.shape)

        rnn_input = torch.cat((dec_input, c), dim=2)
        # print('input: {}'.format(rnn_input))
        s = s.unsqueeze(0)
        h_0 = s
        c_0 = s
        # print('s: {}'.format(s.shape))
        dec_output, (dec_hidden, c_n) = self.rnn(rnn_input, (h_0, c_0))
        # print('dec out: ')
        # print(dec_output.shape)
        # print(dec_hidden.shape)

        dec_input = dec_input.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)

        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, dec_input), dim=1))

        # print(pred.shape)
        dec_hidden =dec_hidden.squeeze(0)
        # print(dec_hidden.shape)
        # return pred, dec_hidden.squeeze(0)
        return pred, dec_hidden


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


class SubSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, timestamp, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size, enc_input_dim]
        # trg = [trg_len, batch_size, dec_output_dim]
        # use teacher forcing
        src = src.to(self.device)
        trg = trg.to(self.device)

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        dec_output_dim = trg.shape[2]
        pre_index, suc_index = get_pre_suc(timestamp, trg_len)

        # trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        # outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        enc_output, s = self.encoder(src)
        # print('$$$$ ')
        # print(type(s))
        # print(s)

        #
        dec_input = src[0]
        # outputs = torch.zeros(trg_len, batch_size, dec_output_dim).to(self.device)
        outputs = torch.zeros(trg_len, batch_size, dec_output_dim)
        outputs = outputs.to(self.device)
        outputs[0] = dec_input

        for t in range(1, trg_len):
            # print('####### {}'.format(t))
            # dec_output, s = self.decoder(dec_input, s, enc_output)
            precursor = pre_index[t]
            successor = suc_index[t]
            # print(type(s))
            # print(s)
            dec_output, s = self.decoder(dec_input, s, enc_output, src[precursor], src[successor])

            # Copy 机制
            if timestamp.count(t) > 0:
                outputs[t] = src[timestamp.index(t)]
            else:
                outputs[t] = dec_output

            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = trg[t] if teacher_force else dec_output
            # 是否使用 teacher forcing
            # teacher_force = random.random() < teacher_forcing_ratio
            # top1 = dec_output.argmax(1)
            # dec_input = trg[t] if teacher_force else top1

        return outputs


