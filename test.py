from time import time

import numpy
import torch
import numpy as np
import multiprocessing
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib.pyplot import MultipleLocator
import datetime

def load_cf(filename):
    user = []
    item = []
    user_dict = dict()

    lines = open(filename, 'r').readlines()
    for l in lines:
        tmp = l.strip()
        print("tmp = {}\n".format(tmp))
        inter = [int(i) for i in tmp.split()]
        print("inter = {}\n".format(inter))
        break
        if len(inter) > 1:
            user_id, item_ids = inter[0], inter[1:]
            item_ids = list(set(item_ids))

            for item_id in item_ids:
                user.append(user_id)
                item.append(item_id)
            user_dict[user_id] = item_ids

    user = np.array(user, dtype=np.int32)
    item = np.array(item, dtype=np.int32)
    return (user, item), user_dict

# file_name = "/home/knowledge303/Documents/zjm/data/metric/add_kg/test_final.txt"
# load_cf(file_name)
# print(4 // 2)

# cf_score = [[0, 0.4, 0.5],
#             [0.2, 0.235, 0.8],
#             [-0.5, -0.7, 5.9]]
# cf_score = torch.tensor(cf_score)
# scores_normalized = torch.sigmoid(cf_score)
# print(scores_normalized)
# scores_normalized[scores_normalized >= 0.5] = 1
# scores_normalized[scores_normalized < 0.5] = 0
# print(scores_normalized)
#
# TP = scores_normalized[scores_normalized == 1].shape[0]
# print(TP)

def tt():
    res = [[1, 2, 3],
           [4, 5, 6]]
    res = numpy.zeros([5, 3])
    # res = res.cuda()
    for i in range(len(res)):
        print(i)
        print(type(res))



def add(x):
    a = x[0]
    b = x[1]
    res = a + b
    return {'res': np.array(res)}



# cores = multiprocessing.cpu_count() // 2
# pool = multiprocessing.Pool(cores)
# a = [1, 2, 3]
# b = [4, 5, 6]
# user_batch_hit_scores = zip(a, b)
# print("multiprocessing ....\n")
# batch_result = pool.map(add, user_batch_hit_scores)
# batch_res = []
# for re in batch_result:
#     batch_res.append(re['res'])
#
# print(batch_res)
# batch_res = np.array(batch_res)
# print(type(batch_res))
# print(batch_res.shape)
# pool.close()

def cal_auc():
    hits = [0, 1, 1, 1, 0]
    scores = [0.4, 0.66, 0.9, 0.8, 0.7]
    fpr, tpr, thresholds = roc_curve(hits, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("auc = {}\nfpr = {}\ntpr = {}\nthresholds = {}".format(auc, fpr, tpr, thresholds))


def draw_auc():
    hits = [0, 1, 1, 1, 0]
    scores = [0.4, 0.66, 0.9, 0.8, 0.7]
    fpr, tpr, thresholds = roc_curve(hits, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    hits2 = [0, 0, 1, 0, 0]
    scores2 = [0.4, 0.66, 0.55, 0.8, 0.7]
    fpr2, tpr2, thresholds2 = roc_curve(hits2, scores2, pos_label=1)
    auc2 = metrics.auc(fpr2, tpr2)

    plt.plot(fpr, tpr, ls="-", lw=1, label="{:s} (AUC={:.4f})".format('cc1', auc))
    plt.plot(fpr2, tpr2, ls="-", lw=1, label="{:s} (AUC={:.4f})".format('cc2', auc2))
    plt.legend(loc='best', fancybox=True)
    x_major_locator = MultipleLocator(0.1)
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    plt.grid(True, linestyle='--')
    # create_dir(os.path.join(result_dir, result_file_prefix + '_ROC'))
    # plt.savefig(os.path.join(result_dir, result_file_prefix + '_ROC', '%d.png' % epoch))
    # c_fig = plt.gcf()  # 'get current figure'
    # c_fig.savefig(os.path.join(result_dir, result_file_prefix + '_ROC', '%d.eps' % epoch), format='eps', dpi=1000)
    plt.show()
    plt.close()


def draw_tsv():
    epoch = list(range(70))
    epoch2 = list(range(50))
    auc = [0.7906, 0.8046, 0.8203, 0.8248, 0.8301, 0.8309, 0.8317, 0.8319, 0.8330,
           0.8335, 0.8325, 0.8326, 0.8330, 0.8335, 0.8341, 0.8334, 0.8334, 0.8323,
           0.8308, 0.8286, 0.8265, 0.8259, 0.8235, 0.8218, 0.8188, 0.8163, 0.8147,
           0.8161, 0.8148, 0.8127, 0.8119, 0.8141, 0.8116, 0.8101, 0.8107, 0.8083,
           0.8112, 0.8114, 0.8122, 0.8115, 0.8093, 0.8108, 0.8101, 0.8105, 0.8121,
           0.8155, 0.8118, 0.8138, 0.8155, 0.8148, 0.8145, 0.8143, 0.8129, 0.8161,
           0.8142, 0.8160, 0.8152, 0.8183, 0.8161, 0.8177, 0.8157, 0.8150, 0.8174,
           0.8153, 0.8155, 0.8136, 0.8162, 0.8159, 0.8165, 0.8168
           ]
    acc = [0.0036, 0.4760, 0.5327, 0.5866, 0.7010, 0.7331, 0.7428, 0.7175, 0.7470,
           0.7567, 0.7405, 0.7507, 0.7877, 0.7586, 0.7562, 0.7882, 0.7785, 0.7913,
           0.7920, 0.7968, 0.8073, 0.7985, 0.8109, 0.8118, 0.8155, 0.8232, 0.8193,
           0.8190, 0.8178, 0.8242, 0.8289, 0.8233, 0.8267, 0.8286, 0.8266, 0.8326,
           0.8287, 0.8318, 0.8307, 0.8289, 0.8363, 0.8321, 0.8363, 0.8376, 0.8329,
           0.8259, 0.8369, 0.8320, 0.8301, 0.8349, 0.8362, 0.8350, 0.8410, 0.8328,
           0.8422, 0.8377, 0.8409, 0.8342, 0.8398, 0.8372, 0.8425, 0.8447, 0.8375,
           0.8441, 0.8468, 0.8492, 0.8450, 0.8479, 0.8425, 0.8418
           ]
    auc2 = [0.7801, 0.8173, 0.8217, 0.8275, 0.8299, 0.8275, 0.8220, 0.8227, 0.8249,
            0.8266, 0.8288, 0.8266, 0.8288, 0.8267, 0.8296, 0.8295, 0.8297, 0.8299,
            0.8299, 0.8316, 0.8291, 0.8320, 0.8310, 0.8314, 0.8306, 0.8311, 0.8306,
            0.8296, 0.8282, 0.8288, 0.8279, 0.8269, 0.8260, 0.8268, 0.8261, 0.8250,
            0.8245, 0.8261, 0.8238, 0.8250, 0.8232, 0.8240, 0.8237, 0.8269, 0.8213,
            0.8221, 0.8206, 0.8232, 0.8201, 0.8199,

        ]
    acc2 = [0.0033, 0.0033, 0.4671, 0.5953, 0.8132, 0.8179, 0.8125, 0.8214, 0.8184,
            0.8050, 0.7898, 0.8048, 0.7878, 0.8161, 0.7970, 0.7953, 0.7969, 0.7956,
            0.8085, 0.8003, 0.8147, 0.7966, 0.8084, 0.8054, 0.8188, 0.8074, 0.8114,
            0.8167, 0.8187, 0.8089, 0.8158, 0.8239, 0.8229, 0.8199, 0.8189, 0.8259,
            0.8220, 0.8105, 0.8192, 0.8165, 0.8233, 0.8172, 0.8233, 0.8149, 0.8262,
            0.8253, 0.8253, 0.8232, 0.8259, 0.8273,
            ]
    # res2 = res.copy()
    # res2.reverse()

    res2 = [0.8169, 0.8348, 0.8344, 0.8183, 0.7911, 0.7882, 0.824, 0.8097, 0.8213, 0.8033, 0.8216, 0.8194]
    print(res2)

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    plt.plot(epoch, auc, ls="-", lw=1, label="with no-consequence relationship")
    plt.plot(epoch2, auc2, ls="-", lw=1, label="without no-consequence relationship")
    plt.legend(loc='best', fancybox=True)

    # auc
    x_major_locator = MultipleLocator(5)
    y_major_locator = MultipleLocator(0.003)
    plt.xlabel(u'epoch')
    plt.ylabel(u'auc')
    plt.title('AUC based on different relationships')

    # acc
    # x_major_locator = MultipleLocator(5)
    # y_major_locator = MultipleLocator(0.05)
    # plt.xlabel(u'epoch')
    # plt.ylabel(u'acc')
    # plt.title('ACC based on different relationships')

    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    plt.grid(True, linestyle='--')
    # create_dir(os.path.join(result_dir, result_file_prefix + '_ROC'))
    # plt.savefig(os.path.join(result_dir, result_file_prefix + '_ROC', '%d.png' % epoch))
    # c_fig = plt.gcf()  # 'get current figure'
    # c_fig.savefig(os.path.join(result_dir, result_file_prefix + '_ROC', '%d.eps' % epoch), format='eps', dpi=1000)
    plt.show()
    plt.close()


def draw_topk():
    topK = [1, 2, 5, 10, 20, 50, 100]
    kgat_precision = [0.025303643, 0.022604588, 0.020107962, 0.016565451, 0.014794198, 0.011356276, 0.008829284]
    kgat_recall = [0.0108609, 0.015779942, 0.027552303, 0.037610695, 0.053204104, 0.080417641, 0.101999342]
    kgat_ndcg = [0.025303644, 0.027301769, 0.032263916, 0.034288681, 0.038309049, 0.045239908, 0.050577895]
    kgat_f1 = [0.011920499, 0.012585096, 0.013704745, 0.013339985, 0.014111565, 0.012691403, 0.010673252]
    # kgat_hit_ratio = []


    plt.plot(topK, kgat_precision, ls="-", lw=1, label="KGAT")
    plt.legend(loc='best', fancybox=True)

    plt.xticks(topK, topK)
    plt.yticks(kgat_precision, kgat_precision)
    x_major_locator = MultipleLocator(100)
    y_major_locator = MultipleLocator(0.005)
    plt.xlabel(u'K')
    plt.ylabel(u'Precision@K')
    plt.title('Precision@K based on different models')

    ax = plt.gca()
    # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    plt.grid(True, linestyle='--')
    # create_dir(os.path.join(result_dir, result_file_prefix + '_ROC'))
    # plt.savefig(os.path.join(result_dir, result_file_prefix + '_ROC', '%d.png' % epoch))
    # c_fig = plt.gcf()  # 'get current figure'
    # c_fig.savefig(os.path.join(result_dir, result_file_prefix + '_ROC', '%d.eps' % epoch), format='eps', dpi=1000)
    plt.show()
    plt.close()



draw_topk()
