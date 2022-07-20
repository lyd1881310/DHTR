import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random


# Decoder 的多层感知机
class MLP(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=512):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size, baise=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size, baise=False)

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

        # 双向 LSTM: 合并最后时刻的隐藏层输出
        # s = [batch_size, dec_hid_dim]
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))

        return enc_output, s


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
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

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
        dec_input = self.fc_in(torch.cat((r, dec_input), dim=1), dec_input)

        a = self.attention(s, enc_output).unsqueeze(1)
        enc_output = enc_output.transpose(0, 1)
        c = torch.bmm(a, enc_output).transpose(0, 1)

        rnn_input = torch.cat((dec_input, c), dim=2)
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))

        dec_input = dec_input.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)

        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, dec_input), dim=1))

        return pred, dec_hidden.squeeze(0)


class SubSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size, enc_input_dim]
        # trg = [trg_len, batch_size, dec_output_dim]
        # use teacher forcing

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        enc_output, s = self.encoder(src)

        #
        dec_input = src[0]

        for t in range(1, trg_len):
            dec_output, s = self.decoder(dec_input, s, enc_output)
            outputs[t] = dec_output

            # 是否使用 teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(1)
            dec_input = trg[t] if teacher_force else top1

        return outputs


