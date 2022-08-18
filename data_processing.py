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
def filt_data(data, x_min, x_max, y_min, y_max):
    # x_min = -8.8
    # x_max = -8.6
    # y_min = 41.10
    # y_max = 41.20
    res = {'index': [], 'traj': []}
    save_path = './data/data_set_5.csv'
    idx = 0
    for index, row in data.iterrows():
        traj = polyline_to_list(row['traj'])
        out_of_bound = False
        for pos in traj:
            if pos[0] < x_min or pos[0] > x_max or pos[1] < y_min or pos[1] > y_max:
                out_of_bound = True
                break
        if not out_of_bound and len(traj) > 8:
            res['index'].append(idx)
            res['traj'].append(traj)
            idx += 1
    print('[{}, {}], [{}, {}], len: {}, filt num: {}'.format(x_min, x_max, y_min, y_max, len(data), len(data) - idx))
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
    save_path = './data/data_set_7.csv'
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
    data_frame = pd.read_csv('./data/data_set_7.csv')
    data = []
    thresh = 200000
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


def get_all_data():
    x_min = -8.68500
    x_max = -8.59766
    y_min = 41.12
    y_max = 41.198378
    y = math.fabs((y_min + y_max) / 2) * math.pi / 180
    dx = 0.1 / (111 * math.cos(y))  # 100m 对应的纬度
    dy = 0.1 / 111
    print(dx)
    print(dy)
    print((x_max - x_min) / dx)
    print((y_max - y_min) / dy)

    data = pd.read_csv('./data/data_set_2.csv')


# 经度 73 格 纬度 87 格
# 获取网格中心坐标
def cell_to_coordinate(x, y):
    x_min = -8.68500
    x_max = -8.59766
    y_min = 41.12
    y_max = 41.198378
    dx = 0.0011966000451121266
    dy = 0.0009009009009009009
    return x_min + dx * x + dx / 2, y_min + dy * y + dy / 2


def cell_to_num(x, y):
    return x * 87 + y


def num_to_cell(n):
    return n // 87, n % 87


if __name__ == '__main__':
    # # get_all_data()
    x_min = -8.68500
    x_max = -8.59766
    y_min = 41.12
    y_max = 41.198378
    # dx = 0.0011966000451121266
    # dy = 0.0009009009009009009
    # print((x_max - x_min) / dx)
    # print((y_max - y_min) / dy)

    data = pd.read_csv('./data/data_set_5.csv')
    # test_data = data[0: 5]
    # res = []
    # for index, row in test_data.iterrows():
    #     traj = polyline_to_list(row['traj'])
    #     t = []
    #     for pos in traj:
    #         x, y = cell_to_coordinate(pos[0], pos[1])
    #         t.append([x, y])
    #     res.append(t)
    # for item in res:
    #     print(item)
    # print(torch.tensor(res, dtype=torch.float))
    # print(data[0: 284100])
    partition_cells(data[0: 800000], x_min, x_max, y_min, y_max)
    # filt_data(data, x_min, x_max, y_min, y_max)

    # width = 0.087340
    #     # height = 0.078378
    #     #
    #     # x_min = -8.72
    #     # x_max = -8.50
    #     # y_min = 41.10
    #     # y_max = 41.20
    #     #
    #     # x = -8.72
    #     # dx = 0.005
    #     # y = 41.10
    #     # dy = 0.005
    #     #
    #     # while x >= x_min and x + width <= x_max:
    #     #     y = 41.10
    #     #     while y >= y_min and y + height <= y_max:
    #     #         filt_data(data, x, x + width, y, y + height)
    #     #         y += dy
    #     #     x += dx

    # [-8.684999999999995, -8.597659999999996], [41.12000000000001, 41.19837800000001], len: 982883, filt



