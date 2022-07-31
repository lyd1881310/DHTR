def get_pre_suc(timestamp, trg_len):
    pre = {}
    suc = {}
    for i in range(0, trg_len):
        if i == 0:
            pre[0] = 0
        else:
            j = 0
            while timestamp[j] < i:
                j = j + 1
            pre[i] = j - 1
    for i in range(0, trg_len):
        if i == trg_len - 1:
            suc[i] = len(timestamp) - 1
        else:
            j = 0
            while timestamp[j] <= i:
                j = j + 1
            suc[i] = j

    return pre, suc


# a = [0, 2, 5]
# b = [0, 1, 2, 3, 4, 5]
# pre, suc = get_pre_suc(a, 6)
# print(pre)
# print(suc)
