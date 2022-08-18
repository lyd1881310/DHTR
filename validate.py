import torch
from train_model import *
from data_processing import *
from DHTR import *


def eval(model, data, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        batch_num = 0
        for x, y, trg_len, timestamp in get_batches(data, 8, 0.7):
            out, res = model(x, y, timestamp)
            y = y.to(device)
            res = res.to(device)
            loss = criterion(res, y)
            print('eval {} batch loss: {}'.format(batch_num, loss.item()))
            batch_num += 1
            epoch_loss += loss.item()

        print('eval loss: {}'.format(epoch_loss / batch_num))


if __name__ == '__main__':
    path = './saved_models/4_model_epoch9.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    # print(model.parameters())

    train_set, val_set, test_set = split_data_set()

    eval(model, val_set, criterion, device)



