import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import datetime
import os
import numpy as np
save_dir = "/home/knowledge303/Documents/zjm/code/KGAT-pytorch/exp_plt/"


def draw_one_auc(auc, fpr, tpr):
    print("auc = {}\nfpr = {}\ntpr = {}".format(auc, fpr, tpr))
    plt.plot(fpr, tpr, ls="-", lw=1, label="{:s} (AUC={:.4f})".format('KGAT', auc))
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
    plt.savefig(os.path.join(save_dir, '%d.png' % str(datetime.datetime.time())))
    # c_fig = plt.gcf()  # 'get current figure'
    # c_fig.savefig(os.path.join(result_dir, result_file_prefix + '_ROC', '%d.eps' % epoch), format='eps', dpi=1000)
    plt.show()
    plt.close()
