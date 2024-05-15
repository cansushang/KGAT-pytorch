import torch
import numpy as np
import multiprocessing
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, roc_curve, auc
from utils.draw_auc import *

def calc_recall(rank, ground_truth, k):
    """
    calculate recall of one example
    """
    return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(hit, k):
    """
    calculate Precision@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)[:k]
    return np.mean(hit)


def acc_at_one_user(x):
    hits_at_one = x[0]
    cf_scores_sorted_at_one = x[1]
    res = accuracy_score(hits_at_one, cf_scores_sorted_at_one)
    print("one acc = {}\n".format(res))
    return res

def acc_at_all_batch(hits, cf_scores_sorted, k):
    """
    author: lord_can
    calculate acc@k
    hits: array, element is binary (0 / 1), 2-dim
    cf_scores_sorted: array[users, items] 降序排序
    eg:
    """
    res = []
    for u in range(len(hits)):
        res.append(accuracy_score(hits[u, :], cf_scores_sorted[u, :]))
    return res


def auc_acc_at_one_user(x):
    hits_at_one = x[0]
    cf_scores_sorted_at_one = x[1]
    try:
        auc = get_auc(hits_at_one, cf_scores_sorted_at_one)
        # fpr, tpr, thresholds = roc_curve(hits_at_one, cf_scores_sorted_at_one, pos_label=1)
        # print("fpr = {}\ntpr = {}".format(fpr, tpr))
        # auc = metrics.auc(fpr, tpr)
        # print("auc = {}\n".format(auc))
        # fpr_avg = np.concatenate(fpr).mean()
        # fpr_avg = np.mean(fpr)
        # tpr_avg = np.mean(tpr)
    except Exception:
        auc = 0.
        # auc, fpr, tpr, fpr_avg, tpr_avg = 0., 0., 0., 0., 0.
    try:
        cf_scores_sorted_at_one = [1 if i >= 0.5 else 0 for i in cf_scores_sorted_at_one]
        acc = accuracy_score(hits_at_one, cf_scores_sorted_at_one)
    except Exception:
        acc = 0.
    # return auc, acc
    return {'auc': np.array(auc), 'acc': np.array(acc)}
    # return {'auc': np.array(auc), 'fpr': np.array(fpr_avg), 'tpr': np.array(tpr_avg), 'acc': np.array(acc)}


def auc_at_all_batch(hits, cf_scores_sorted, k):
    """
    author: lord_can
    calculate auc@k
    hits: array, element is binary (0 / 1), 2-dim
    cf_scores_sorted: array[users, items] 降序排序
    eg:
    """
    res = []
    # hits = hits.cpu()
    # cf_scores_sorted = cf_scores_sorted.cpu()
    for u in range(len(hits)):
        # res.append(roc_auc_score(hits[u, :k], cf_scores_sorted[u, :k]))
        # res.append(get_auc(hits[u, :k], cf_scores_sorted[u, :k]))

        try:
            # temp = get_auc(hits[u, :k], cf_scores_sorted[u, :k])
            temp = get_auc(hits[u, :], cf_scores_sorted[u, :])
        except Exception:
            temp = 0.
        res.append(temp)

    return res


def get_auc(labels, preds):
    # 这段代码基本上是沿着公式计算的：
    # 1. 先求正样本的rank和
    # 2. 再减去（m*(m+1)/2）
    # 3. 最后除以组合个数

    # 但是要特别注意，需要对预测值pred相等的情况进行了一些处理。
    # 对于这些预测值相等的样本，它们对应的rank是要取平均的

    # 先将data按照pred进行排序
    sorted_data = sorted(list(zip(labels, preds)), key=lambda item: item[1])
    pos = 0.0  # 正样本个数
    neg = 0.0  # 负样本个数
    auc = 0.0
    # 注意这里的一个边界值，在初始时我们将last_pre记为第一个数，那么遍历到第一个数时只会count++
    # 而不会立刻向结果中累加（因为此时count==0，根本没有东西可以累加）
    last_pre = sorted_data[0][1]
    count = 0.0
    pre_sum = 0.0  # 当前位置之前的预测值相等的rank之和，rank是从1开始的，所以在下面的代码中就是i+1
    pos_count = 0.0  # 记录预测值相等的样本中标签是正的样本的个数

    # 为了处理这些预测值相等的样本，我们这里采用了一种lazy计算的策略：
    # 当预测值相等时仅仅累加count，直到下次遇到一个不相等的值时，再将他们一起计入结果
    for i, (label, pred) in enumerate(sorted_data):
        # 注意：rank就是i+1
        if label > 0:
            pos += 1
        else:
            neg += 1
        if last_pre != pred:  # 当前的预测概率值与前一个值不相同
            # lazy累加策略被触发，求平均并计入结果，各个累积状态置为初始态
            auc += pos_count * pre_sum / count  # 注意这里只有正样本的部分才会被累积进结果
            count = 1
            pre_sum = i + 1  # 累积rank被清空，更新为当前数rank
            last_pre = pred
            if label > 0:
                pos_count = 1  # 如果当前样本是正样本 ，则置为1
            else:
                pos_count = 0  # 反之置为0
        # 如果预测值是与前一个数相同的，进入累积状态
        else:
            pre_sum += i + 1  # rank被逐渐累积
            count += 1  # 计数器也被累计
            if label > 0:  # 这里要另外记录正样本数，因为负样本在计算平均
                pos_count += 1  # rank的时候会被计入，但不会被计入rank和的结果

    # 注意这里退出循环后我们要额外累加一下。
    # 这是因为我们上面lazy的累加策略，导致最后一组数据没有累加
    auc += pos_count * pre_sum / count
    auc -= pos * (pos + 1) / 2  # 减去正样本在正样本之前的情况即公式里的(m+1)m/2
    auc = auc / (pos * neg)  # 除以总的组合数即公式里的m*n
    return auc


def hit_ratio_at_k_batch(hits, k):
    """
    author: lord_can
    calculate Hit Ratio@k
    hits: array, element is binary (0 / 1), 2-dim
    eg: 三个用户在测试集中的商品个数分别是10，12，8，
    模型得到的top-10推荐列表中，分别有6个，5个，4个在测试集中，
    那么此时HR的值是   (6+5+4)/(10+12+8) = 0.5  。
    """
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res


def f1_at_k_batch(precision, recall):
    """
    author: lord_can
    calculate f1@k
    precision: array, shape = [1, len(userids)]
    recall: array, shape = [1, len(userids)]
    """
    res = []
    for pre, cal in zip(precision, recall):
        res.append(F1(pre, cal))
    return res


def precision_at_k_batch(hits, k):
    """
    calculate Precision@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = hits[:, :k].mean(axis=1)
    return res


def average_precision(hit, cut):
    """
    calculate average precision (area under PR curve)
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)
    precisions = [precision_at_k(hit, k + 1) for k in range(cut) if len(hit) >= k]
    if not precisions:
        return 0.
    return np.sum(precisions) / float(min(cut, np.sum(hit)))


def dcg_at_k(rel, k):
    """
    calculate discounted cumulative gain (dcg)
    rel: list, element is positive real values, can be binary
    """
    rel = np.asfarray(rel)[:k]
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return dcg


def ndcg_at_k(rel, k):
    """
    calculate normalized discounted cumulative gain (ndcg)
    rel: list, element is positive real values, can be binary
    """
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg


def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = np.inf
    ndcg = (dcg / idcg)
    return ndcg


def recall_at_k(hit, k, all_pos_num):
    """
    calculate Recall@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def recall_at_k_batch(hits, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res


def F1(pre, rec):
    if pre.all() + rec.all() > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


# def calc_auc(ground_truth, prediction):
#     try:
#         res = roc_auc_score(y_true=ground_truth, y_score=prediction)
#     except Exception:
#         res = 0.
#     return res


def logloss(ground_truth, prediction):
    logloss = log_loss(np.asarray(ground_truth), np.asarray(prediction))
    return logloss


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, Ks):
    """
    cf_scores: (n_users, n_items)
    """
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    for idx, u in enumerate(user_ids):
        train_pos_item_list = train_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        cf_scores[idx][train_pos_item_list] = -np.inf
        test_pos_item_binary[idx][test_pos_item_list] = 1

    try:
        cf_scores_sorted, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
    except:
        cf_scores_sorted, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(user_ids)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)

    # print("\nbinary_hit:{}\n".format(binary_hit))
    # print("\ncf_scores_sorted:{}\n".format(cf_scores_sorted))

    metrics_dict = {}
    # print("\nks eval starting....\n")
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['precision'] = precision_at_k_batch(binary_hit, k)
        metrics_dict[k]['recall'] = recall_at_k_batch(binary_hit, k)
        metrics_dict[k]['ndcg'] = ndcg_at_k_batch(binary_hit, k)
        metrics_dict[k]['f1'] = f1_at_k_batch(metrics_dict[k]['precision'], metrics_dict[k]['recall'])
        metrics_dict[k]['hit_ratio'] = hit_ratio_at_k_batch(binary_hit, k)
        # print("\nhit ratio type:\n{}".format(type(metrics_dict[k]['hit_ratio'])))
        # metrics_dict[k]['auc'] = auc_at_k_batch(binary_hit, cf_scores_sorted, k)
        # print("\nauc type:\n{}".format(type(metrics_dict[k]['auc'])))
        # print("\nauc:\n{}".format(metrics_dict[k]['auc']))

    # print("auc & acc is calculating ...\n")
    cores = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(cores)
    cf_scores_sorted = cf_scores_sorted.cpu()
    user_batch_hit_scores = zip(binary_hit, cf_scores_sorted)
    # print("multiprocessing ....\n")
    batch_result = pool.map(auc_acc_at_one_user, user_batch_hit_scores)
    # batch_auc, batch_acc = pool.map(auc_acc_at_one_user, user_batch_hit_scores)
    # batch_acc = pool.map(acc_at_one_user, user_batch_hit_scores)

    # K_min = max(Ks)
    # metrics_dict[K_min]['auc'] = auc_at_all_batch(binary_hit, cf_scores_sorted, k)
    # print("auc = \n{}".format(metrics_dict[K_min]['auc']))
    # metrics_dict[K_min]['acc'] = auc_at_all_batch(binary_hit, cf_scores_sorted, k)
    # print("acc = \n{}".format(metrics_dict[K_min]['acc']))

    batch_auc = []
    batch_acc = []
    # batch_fpr = []
    # batch_tpr = []

    for re in batch_result:
        batch_auc.append(re['auc'])
        batch_acc.append(re['acc'])
        # batch_fpr.append(re['fpr'])
        # batch_tpr.append(re['tpr'])
        # metrics_dict[0]['auc'] += re['auc']
        # metrics_dict[0]['acc'] += re['acc']
    # print("auc = {}\nacc = {}\n".format(batch_auc, batch_acc))
    # print("copying....\n")
    # draw_one_auc(np.mean(batch_auc), batch_fpr, batch_tpr)

    # copy to metrics_dict in all k
    for k in Ks:
        metrics_dict[k]['auc'] = batch_auc
        metrics_dict[k]['acc'] = batch_acc
    # print("copy done\n")

    pool.close()
    return metrics_dict

