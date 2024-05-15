import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib.pyplot import MultipleLocator
import datetime
import sys
import math


class Paint(object):

    def __init__(self):
        self.auc_data_kgcn = []
        self.acc_data_kgcn = []
        self.precision_data_kgcn = []
        self.recall_data_kgcn = []
        self.f1_data_kgcn = []
        self.auc_data_kni = []
        self.acc_data_kni = []
        self.f1_data_kni = []
        self.auc_data_ckan = []
        self.acc_data_ckan = []
        self.f1_data_ckan = []
        self.auc_data_7 = []
        self.acc_data_7 = []
        self.auc_data_concat = []
        self.acc_data_concat = []
        self.auc_data_nei = []
        self.acc_data_nei = []


        self.auc_data_kgat = []
        self.auc_data_cke = []
        self.auc_data_ecfkg = []
        self.auc_data_bprmf = []

        self.acc_data_kgat = []
        self.acc_data_cke = []
        self.acc_data_ecfkg = []
        self.acc_data_bprmf = []
        self.acc_data_kgat_seven_kg = []
    @staticmethod
    def loadfile(filename):
        """ load a file, return a generator. """

        # 以只读的方式打开传入的文件
        fp = open(filename, 'r', encoding='utf-8')
        # enumerate()为枚举，i为行号从0开始，line为值
        for i, line in enumerate(fp, 1):
            # yield 迭代去下一个值，类似next()
            # line.strip()用于去除字符串头尾指定的字符。
            yield line.strip('\r\n')
            # 计数
            if i % 10000000 == 0:
                print('loading %s(%s)' % (filename, i), file=sys.stderr)
        # 关闭文件
        fp.close()

    def get_data(self, filename, model, agg, flag):
        i = 0
        for line in self.loadfile(filename):

            # _, precision, _, recall, _, acc, _, auc, _, f1, _ = line.split(',')
            _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, \
            auc, acc = line.split('\t')
            try:
                float(auc)
                float(acc)
            except ValueError:
                continue
            if model == 'kgat':
                self.auc_data_kgat.append(float(auc))
                self.acc_data_kgat.append(float(acc))
            if model == 'cke':
                self.auc_data_cke.append(float(auc))
                self.acc_data_cke.append(float(acc))
            if model == 'bprmf':
                self.auc_data_bprmf.append(float(auc))
                self.acc_data_bprmf.append(float(acc))
            if model == 'ecfkg':
                self.auc_data_ecfkg.append(float(auc))
                self.acc_data_ecfkg.append(float(acc))
            if model == 'kgcn':
                self.precision_data_kgcn.append(float(precision))
                self.recall_data_kgcn.append(float(recall))
                self.acc_data_kgcn.append(float(acc))
                self.auc_data_kgcn.append(float(auc))
                self.f1_data_kgcn.append(float(f1))
            elif model == 'kni':
                self.acc_data_kni.append(float(acc))
                self.auc_data_kni.append(float(auc))
                self.f1_data_kni.append(float(f1))
            elif model == 'ckan':
                self.auc_data_ckan.append(float(auc))
                self.f1_data_ckan.append(float(f1))

            if flag == "8":
                self.acc_data_kgcn.append(float(acc))
                self.auc_data_kgcn.append(float(auc))
                if len(self.auc_data_kgcn) == 100:
                    break
            elif flag == "7":
                self.acc_data_7.append(float(acc))
                self.auc_data_7.append(float(auc))
                if len(self.auc_data_7) == 100:
                    break

            if agg == "sum":
                self.acc_data_kgcn.append(float(acc))
                self.auc_data_kgcn.append(float(auc))
                if len(self.auc_data_kgcn) == 100:
                    break
            elif agg == "concat":
                self.acc_data_concat.append(float(acc))
                self.auc_data_concat.append(float(auc))
                if len(self.auc_data_concat) == 100:
                    break
            elif agg == "nei":
                self.acc_data_nei.append(float(acc))
                self.auc_data_nei.append(float(auc))
                if len(self.auc_data_nei) == 100:
                    break

        if model == 'kgcn':
            list_temp = []
            for i in range(len(self.auc_data_kgcn)):
                if i % 10 == 0:
                    list_temp.append(self.auc_data_kgcn[i])
            self.auc_data_kgcn = list_temp
            list_temp = []
            for i in range(len(self.f1_data_kgcn)):
                if i % 10 == 0:
                    list_temp.append(self.f1_data_kgcn[i])
            self.f1_data_kgcn = list_temp

        elif model == 'kni':
            list_temp = []
            for i in range(len(self.auc_data_kni)):
                if len(list_temp) == 20:
                    continue
                if i % 2 == 0:
                    list_temp.append(self.auc_data_kni[i])
            self.auc_data_kni = list_temp
            list_temp = []
            for i in range(len(self.f1_data_kni)):
                if len(list_temp) == 20:
                    continue
                if i % 2 == 0:
                    list_temp.append(self.f1_data_kni[i])
            self.f1_data_kni = list_temp

        # print(self.precision_data_kgcn)
        # print(self.recall_data_kgcn)
        # print(self.acc_data_kgcn)

        print(self.auc_data_kgcn)
        # print(self.auc_data_7)
        print(self.auc_data_kni)
        print(self.auc_data_ckan)
        # print(self.f1_data_kgcn)

    def draw_base_auc(self):

        epoch = list(range(20))
        plt.plot(epoch, self.auc_data_kgcn, ls="-", lw=1, label="model is kgcn")
        plt.plot(epoch, self.auc_data_kni, ls="-", lw=1, label="model is kni")
        plt.plot(epoch, self.auc_data_ckan, ls="-", lw=1, label="model is ckan")
        plt.legend(loc='best', fancybox=True)

        # auc
        x_major_locator = MultipleLocator(2)
        y_major_locator = MultipleLocator(0.007)
        plt.xlabel(u'epoch')
        plt.ylabel(u'auc')
        plt.title('AUC')

        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(y_major_locator)
        plt.grid(True, linestyle='--')

        plt.show()
        plt.close()

    def draw_base_acc2(self):

        epoch = list(range(70))
        plt.plot(epoch, self.acc_data_kgat_seven_kg, ls="-", lw=1, label="without no-consequence relationship")
        plt.plot(epoch, self.acc_data_kgat, ls="-", lw=1, label="with no-consequence relationship")
        plt.legend(loc='best', fancybox=True)

        # auc
        x_major_locator = MultipleLocator(5)
        y_major_locator = MultipleLocator(0.1)
        plt.xlabel(u'epoch')
        plt.ylabel(u'acc')
        plt.title('ACC based on different relationships')

        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(y_major_locator)
        plt.grid(True, linestyle='--')

        plt.show()
        plt.close()

    def draw_base_auc2(self):
        epoch = list(range(70))
        plt.plot(epoch, self.auc_data_kgat, ls="-", lw=1, label="KGAT")
        plt.plot(epoch, self.auc_data_cke, ls="-", lw=1, label="CKE")
        plt.plot(epoch, self.auc_data_bprmf, ls="-", lw=1, label="BPRMF")
        plt.plot(epoch, self.auc_data_ecfkg, ls="-", lw=1, label="ECFKG")
        # plt.plot(epoch, self.auc_data_kni, ls="-", lw=1, label="model is kni")
        # plt.plot(epoch, self.auc_data_ckan, ls="-", lw=1, label="model is ckan")
        plt.legend(loc='best', fancybox=True)

        # auc
        x_major_locator = MultipleLocator(5)
        y_major_locator = MultipleLocator(0.02)
        plt.xlabel(u'epoch')
        plt.ylabel(u'auc')
        plt.title('AUC based on different relationships')

        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(y_major_locator)
        plt.grid(True, linestyle='--')

        plt.show()
        plt.close()

    def draw_base_acc(self):

        epoch = list(range(100))
        plt.plot(epoch, self.acc_data_kgcn, ls="-", lw=1, label="agg is sum")
        plt.plot(epoch, self.acc_data_concat, ls="-", lw=1, label="agg is concat")
        plt.plot(epoch, self.acc_data_nei, ls="-", lw=1, label="agg is neighbor")
        plt.legend(loc='best', fancybox=True)

        # acc
        x_major_locator = MultipleLocator(10)
        y_major_locator = MultipleLocator(0.005)
        plt.xlabel(u'epoch')
        plt.ylabel(u'acc')
        plt.title('ACC')

        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(y_major_locator)
        plt.grid(True, linestyle='--')

        plt.show()
        plt.close()

    def draw_base_f1(self):

        epoch = list(range(20))
        plt.plot(epoch, self.f1_data_kgcn, ls="-", lw=1, label="model is kgcn")
        plt.plot(epoch, self.f1_data_kni, ls="-", lw=1, label="model is kni")
        plt.plot(epoch, self.f1_data_ckan, ls="-", lw=1, label="model is ckan")
        plt.legend(loc='best', fancybox=True)

        # f1
        x_major_locator = MultipleLocator(2)
        y_major_locator = MultipleLocator(0.007)
        plt.xlabel(u'epoch')
        plt.ylabel(u'f1')
        plt.title('F1')

        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(y_major_locator)
        plt.grid(True, linestyle='--')

        plt.show()
        plt.close()

    def draw_base_precision(self):

        epoch = list(range(200))
        plt.plot(epoch, self.precision_data_kgcn, ls="-", lw=1, label="precision")
        # plt.plot(epoch2, auc2, ls="-", lw=1, label="without no-consequence relationship")
        plt.legend(loc='best', fancybox=True)

        # precision
        x_major_locator = MultipleLocator(20)
        y_major_locator = MultipleLocator(0.003)
        plt.xlabel(u'epoch')
        plt.ylabel(u'precision')
        plt.title('Precision')

        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(y_major_locator)
        plt.grid(True, linestyle='--')

        plt.show()
        plt.close()

    def draw_base_recall(self):

        epoch = list(range(200))
        plt.plot(epoch, self.recall_data_kgcn, ls="-", lw=1, label="recall")
        # plt.plot(epoch2, auc2, ls="-", lw=1, label="without no-consequence relationship")
        plt.legend(loc='best', fancybox=True)

        # recall
        x_major_locator = MultipleLocator(20)
        y_major_locator = MultipleLocator(0.003)
        plt.xlabel(u'epoch')
        plt.ylabel(u'recall')
        plt.title('Recall')

        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(y_major_locator)
        plt.grid(True, linestyle='--')

        plt.show()
        plt.close()

    def draw_auc_kgcn_and_kni(self):

        epoch = list(range(200))
        plt.plot(epoch, self.auc_data_kgcn, ls="-", lw=1, label="auc")
        # plt.plot(epoch2, auc2, ls="-", lw=1, label="without no-consequence relationship")
        plt.legend(loc='best', fancybox=True)

        # auc
        x_major_locator = MultipleLocator(20)
        y_major_locator = MultipleLocator(0.003)
        plt.xlabel(u'epoch')
        plt.ylabel(u'auc')
        plt.title('AUC')

        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(y_major_locator)
        plt.grid(True, linestyle='--')

        plt.show()
        plt.close()


    def draw_topk(self, filename):
        for line in self.loadfile(filename):
            # _, precision, _, recall, _, acc, _, auc, _, f1, _ = line.split(',')
            _ = 0
            precision_1, recall_1, ndcg_1, f1_1, hit_ratio_1 = 0, 0, 0, 0, 0
            precision_2, recall_2, ndcg_2, f1_2, hit_ratio_2 = 0, 0, 0, 0, 0
            precision_5, recall_5, ndcg_5, f1_5, hit_ratio_5 = 0, 0, 0, 0, 0
            precision_10, recall_10, ndcg_10, f1_10, hit_ratio_10 = 0, 0, 0, 0, 0
            precision_20, recall_20, ndcg_20, f1_20, hit_ratio_20 = 0, 0, 0, 0, 0
            precision_50, recall_50, ndcg_50, f1_50, hit_ratio_50 = 0, 0, 0, 0, 0
            precision_100, recall_100, ndcg_100, f1_100, hit_ratio_100 = 0, 0, 0, 0, 0
            auc, acc = 0, 0

            _, precision_1, recall_1, ndcg_1, f1_1, hit_ratio_1, precision_2, recall_2, ndcg_2, f1_2, hit_ratio_2, precision_5, recall_5, ndcg_5, f1_5, hit_ratio_5, precision_10, recall_10, ndcg_10, f1_10, hit_ratio_10, precision_20, recall_20, ndcg_20, f1_20, hit_ratio_20, precision_50, recall_50, ndcg_50, f1_50, hit_ratio_50, precision_100, recall_100, ndcg_100, f1_100, hit_ratio_100, auc, acc = line.split('\t')
            print(max(auc))
            try:
                float(auc)
                float(acc)
            except ValueError:
                continue

        return 1


if __name__ == '__main__':
    paint = Paint()
    # paint.get_data('../data/test_eval_tf_utf8.csv', model='kgcn', agg=False, flag=False)
    # paint.get_data('../data/test_KNI_tf_utf8.csv', model='kni', agg=False, flag=False)
    # paint.get_data('../data/test_CKAN.csv', model='ckan', agg=False, flag=False)
    # paint.get_data('./trained_model/KGAT/预处理后数据/embed-dim8_relation-dim8_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/metrics_bi.tsv', model='kgat', agg=False, flag=False)
    # paint.get_data('C:/Users/Dlian/Desktop/毕设/结果/KGCN/eval/zjm/ckemetrics.tsv', model='cke', agg=False, flag=False)
    # paint.get_data('C:/Users/Dlian/Desktop/毕设/结果/KGCN/eval/zjm/bprmfmetrics.tsv', model='bprmf', agg=False, flag=False)
    # paint.get_data('C:/Users/Dlian/Desktop/毕设/结果/KGCN/eval/zjm/ecfkgmetrics.tsv', model='ecfkg', agg=False, flag=False)
    # paint.draw_base_auc2()
    # paint.draw_base_acc2()
    # paint.draw_base_acc()
    # paint.draw_base_f1()
    # paint.draw_base_precision()
    # paint.draw_base_recall()
    fp = open('./home/knowledge303/Documents/zjm/res/kgat_8kg_70epoch_bi/embed-dim8_relation-dim8_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/metrics.csv', 'r', encoding='utf-8')
    # paint.get_data('./trained_model/exp_res/kgat_7kg_70epoch_bi/embed-dim8_relation-dim8_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/metrics.csv', model='kgat_7kg', agg=False, flag=False)
    # paint.get_data('/home/knowledge303/Documents/zjm/res/kgat_8kg_70epoch_bi/embed-dim8_relation-dim8_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/metrics.csv', model='kgat_8kg', agg=False, flag=False)
    paint.draw_base_acc2()

