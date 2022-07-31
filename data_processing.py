import pandas as pd
import re
import math
import torch


def polyline_to_list(ori_str):
    str = ori_str.replace('[', '').replace(']', '')
    traj = str.split(',')
    res = []
    if len(traj) % 2 != 0:
        print('Wrong polyline')
        print(ori_str)
        return res
    i = 0
    while i < len(traj):
        point = []
        point.append(float(traj[i]))
        point.append(float(traj[i + 1]))
        res.append(point)
        i += 2
    return res


def build_data_set_1(tp):
    raw_path = './raw_data/{}.csv'.format(tp)
    save_path = './data/{}_set_1.csv'.format(tp)
    raw_data = pd.read_csv(raw_path)
    print(type(raw_data['MISSING_DATA'][5]))
    raw_data = raw_data[raw_data['MISSING_DATA'] == False]
    res = {'index': [], 'traj': []}

    idx = 0
    null_index = []
    for index, row in raw_data.iterrows():
        if row['POLYLINE'] == '[]':
            null_index.append(index)
            continue
        traj = polyline_to_list(row['POLYLINE'])
        res['index'].append(idx)
        res['traj'].append(traj)
        idx += 1

    print(len(res['traj']))
    print(null_index)
    print(len(null_index))
    res = pd.DataFrame(res)
    res.to_csv(save_path)


def get_bound(df):
    inf = 10000.1
    x_min = inf
    x_max = -inf
    y_min = inf
    y_max = -inf
    for index, row in df.iterrows():
        traj = polyline_to_list(row['traj'])
        for pos in traj:
            # print(type(pos))
            # print(type(pos[0]))
            x_min = min(x_min, pos[0])
            x_max = max(x_max, pos[0])
            y_min = min(y_min, pos[1])
            y_max = max(y_max, pos[1])
    print('x_min:{}, x_max:{}, y_min:{}, y_max:{}'.format(x_min, x_max, y_min, y_max))


# 删除超出指定经纬度范围的轨迹
def filt_data(data):
    x_min = -8.8
    x_max = -8.6
    y_min = 41.10
    y_max = 41.20
    res = {'index': [], 'traj': []}
    save_path = './data/data_set_2.csv'
    idx = 0
    for index, row in data.iterrows():
        traj = polyline_to_list(row['traj'])
        out_of_bound = False
        for pos in traj:
            if pos[0] < x_min or pos[0] > x_max or pos[1] < y_min or pos[1] > y_max:
                out_of_bound = True
                break
        if not out_of_bound:
            res['index'].append(idx)
            res['traj'].append(traj)
            idx += 1
    print('filt num: {}'.format(idx))
    res = pd.DataFrame(res)
    res.to_csv(save_path)


# 划分 cell
def partition_cells(src_data, x1, x2, y1, y2):
    '''

    :param src_data:
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :return:
    1维度 = 111km
    1精度 = 111 cos \theta
    '''
    save_path = './data/data_cell_3.csv'
    y = math.fabs((y1 + y2) / 2) * math.pi / 180
    print('y: {}, cosy: {}'.format(y, math.cos(y)))
    dx = 0.1 / (111 * math.cos(y))  # 100m 对应的纬度
    dy = 0.1 / 111
    print('dx: {}, dy: {}'.format(dx, dy))
    res = {'traj': []}
    for index, row in src_data.iterrows():
        tra = []
        traj = polyline_to_list(row['traj'])
        for point in traj:
            lx = int((point[0] - x1) / dx)
            ly = int((point[1] - y1) / dy)
            tra.append([lx, ly])
        res['traj'].append(tra)
    res = pd.DataFrame(res)
    res.to_csv(save_path)


# 删除长度小于 seq_len 的序列
def del_short_seq(data, seq_len):
    save_path = './data/data_set_4.csv'
    res = {'traj': []}
    for index, row in data.iterrows():
        traj = polyline_to_list(row['traj'])
        if len(traj) >= seq_len:
            res['traj'].append(traj)

    print('len: {}'.format(len(res)))
    res = pd.DataFrame(res)
    res.to_csv(save_path)

# train:
# x_min:-36.913779, x_max:52.900803, y_min:31.992111, y_max:51.037119
# test:
# x_min:-8.729766, x_max:-7.542072, y_min:41.062563, y_max:41.562351


def split_data_set():
    data_frame = pd.read_csv('./data/data_set_4.csv')
    data = []
    thresh = 300000
    cnt = 0
    for index, row in data_frame.iterrows():
        data.append(polyline_to_list(row['traj']))
        cnt += 1
        if cnt >= thresh:
            break
    # print(data[0: 10])
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.1)
    test_size = len(data) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset=data, lengths=[train_size, val_size, test_size])
    return list(train_set), list(val_set), list(test_set)


if __name__ == '__main__':
    pass
    # data = [
    #     [[1, 1], [1, 1], [1, 1]],
    #     [[2, 2], [2.4, 2]],
    #     [[3, 3], [3, 3]],
    #     [[4, 4], [4, 4]],
    #     [[5, 5], [5, 5]],
    #     [[6, 6], [6, 6]],
    # ]
    # a, b, c = torch.utils.data.random_split(data, [3, 2, 1])
    # a = list(a)
    # b = list(b)
    # c = list(c)
    # print(a)
    # print(b)
    # print(c)


    # a, b, c = split_data_set()
    # print(type(a))
    # print(type(b))
    # print(type(c))
    # print(len(a))
    # print(len(b))
    # print(len(c))
    # print(a[5])
    # df = pd.read_csv('./data/data_cell_3.csv')
    # # filt_data(df)
    # x_min = -8.8
    # x_max = -8.6
    # y_min = 41.10
    # y_max = 41.20
    # # partition_cells(df, x_min, x_max, y_min, y_max)
    # del_short_seq(df, 8)

    # build_data_set_1('train')
    # x_min = -8.729766
    # x_max = -7.542072
    # y_min = 41.062563
    # y_max = 41.562351

    # x_min = -8.8
    # x_max = -8.7
    # y_min = 41.10
    # y_max = 41.20
    #
    #
    # # get_bound(df)
    # out_num = 0
    # for index, row in df.iterrows():
    #     traj = polyline_to_list(row['traj'])
    #     for pos in traj:
    #         if pos[0] < x_min or pos[0] > x_max or pos[1] < y_min or pos[1] > y_max:
    #             out_num += 1
    #             break
    #
    # print(len(df['traj']))
    # print(out_num)

    # for index, row in raw_data.iterrows():
    #     if row['MISSING_DATA'] == True:
    #         print('missing: {}, {}'.format(index, row['POLYLINE']))
    #     elif row['POLYLINE'] == '[]':
    #         print('is missing: {}, {}, {}'.format(index, row['MISSING_DATA'], row['POLYLINE']))
    #
    #     if index == 5000:
    #         break



