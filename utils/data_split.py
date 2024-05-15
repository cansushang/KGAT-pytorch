# 不洗牌模式下数据划分情况
import numpy as np
from sklearn.model_selection import KFold
x = np.arange(46).reshape(23,2)
print(x)
kf = KFold(n_splits=5, shuffle=False)
for train_index, test_index in kf.split(x):
    print(train_index, test_index)
# [ 5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22] [0 1 2 3 4]
# [ 0  1  2  3  4 10 11 12 13 14 15 16 17 18 19 20 21 22] [5 6 7 8 9]
# [ 0  1  2  3  4  5  6  7  8  9 15 16 17 18 19 20 21 22] [10 11 12 13 14]
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 19 20 21 22] [15 16 17 18]
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18] [19 20 21 22]