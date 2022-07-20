import pandas as pd
import re


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

# train:
# x_min:-36.913779, x_max:52.900803, y_min:31.992111, y_max:51.037119
# test:
# x_min:-8.729766, x_max:-7.542072, y_min:41.062563, y_max:41.562351


if __name__ == '__main__':
    # build_data_set_1('train')
    x_min = -8.729766
    x_max = -7.542072
    y_min = 41.062563
    y_max = 41.562351
    df = pd.read_csv('./data/train_set_1.csv')
    # get_bound(df)
    in_num = 0
    for index, row in df.iterrows():
        traj = polyline_to_list(row['traj'])
        for pos in traj:
            if pos[0] >= x_min and pos[0] <= x_max and pos[1] >= y_min and pos[1] <= y_max:
                in_num += 1
                break

    print(len(df['traj']))
    print(in_num)

    # for index, row in raw_data.iterrows():
    #     if row['MISSING_DATA'] == True:
    #         print('missing: {}, {}'.format(index, row['POLYLINE']))
    #     elif row['POLYLINE'] == '[]':
    #         print('is missing: {}, {}, {}'.format(index, row['MISSING_DATA'], row['POLYLINE']))
    #
    #     if index == 5000:
    #         break



