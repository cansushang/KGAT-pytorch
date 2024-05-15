import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch.utils.data as data

data_dir = "/home/knowledge303/Documents/zjm/data/metric/add_kg"
data_dir = "/home/knowledge303/Documents/zjm/data/final_data/final_data/预处理后数据/"
data_name = "ratings_final.txt"
target_file = os.path.join(data_dir, "ratings_process.txt")
target_train_file = os.path.join(data_dir, "train.txt")
target_test_file = os.path.join(data_dir, "test.txt")


def data_process(data_path, target_file):
    txt = pd.read_csv(data_path, sep="\t", names=['userId', 'itemId', 'ratings'])
    txt = pd.DataFrame(txt)
    txt = txt[txt['ratings'] == 1]
    cur_u = -1
    res = []
    temp = []
    temp2 = []
    print("data_processing....")
    for indexs in tqdm(txt.index):
        u = txt.loc[indexs].values[0]
        i = txt.loc[indexs].values[1]
        # r = txt.loc[indexs].values[2]
        if cur_u != u:
            temp.append(temp2)
            res.append(temp)
            temp = []
            temp2 = []
            temp.append(u)
            temp2.append(i)
            cur_u = u
        else:
            temp2.append(i)
    f = open(target_file, 'w+')
    del res[0]
    res_str = str(res)
    res_str = res_str.replace("[", "")
    res_str = res_str.replace(",", "")
    res_str = res_str.replace("] ", "\r\n")
    res_str = res_str.replace("]]", "")
    res_str = res_str.replace("]", "")
    f.write(res_str)
    f.close()
    return res



def data_split(res, target_train_file, target_test_file):
    # print(res)
    # print("**********************")
    # for i in range(len(res)):
    #     print(res[i][1])
    print("spliting to train and test...")
    # kf = KFold(n_splits=2, shuffle=False)
    # for train_index, test_index in kf.split(res[0][1]):
        # train_set = res[train_index]
        # test_set = res[test_index]
        # print(train_index, test_index)
        # print("train_set: \n{}".format(train_set))
        # print("test_set: \n{}".format(test_set))
    test_ratio = 0.2
    np.random.seed(555)
    target_train_file = open(target_train_file, 'w+')
    target_test_file = open(target_test_file, 'w+')
    for i in tqdm(range(len(res))):
        # print("current_i = {}".format(i))
        sub_len = len(res[i][1])
        if sub_len < 5:
            continue
        # print("sub_len = {}".format(sub_len))
        shuffled_indices = np.random.permutation(sub_len)
        test_set_size = int(sub_len * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]

        # print("train_indices:\n{}test_indices:\n{}".format(train_indices, test_indices))
        train_set = []
        test_set = []
        train_set_content = []
        test_set_content = []
        train_set.append(res[i][0])
        test_set.append(res[i][0])
        for j in train_indices:
            train_set_content.append(res[i][1][j])
        for k in test_indices:
            test_set_content.append(res[i][1][k])
        train_set.append(train_set_content)
        test_set.append(test_set_content)
        # print("train_set:\n{}\ntest_set:\n{}".format(train_set, test_set))
        train_set_str = str(train_set)
        test_set_str = str(test_set)
        train_set_str = train_set_str.replace("[", "")
        test_set_str = test_set_str.replace("[", "")
        train_set_str = train_set_str.replace(",", "")
        test_set_str = test_set_str.replace(",", "")
        train_set_str = train_set_str.replace("]]", "\r\n")
        test_set_str = test_set_str.replace("]]", "\r\n")
        target_train_file.write(train_set_str)
        target_test_file.write(test_set_str)
    target_train_file.close()
    target_test_file.close()



if __name__ == '__main__':

    res = data_process(os.path.join(data_dir, data_name), target_file)
    data_split(res, target_train_file, target_test_file)
    # test()