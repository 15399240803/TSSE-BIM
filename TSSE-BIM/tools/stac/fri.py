import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# def friedman(n,k,rank_matrix):
#     sumr = sum(list(map(lambda x:np.mean(x) ** 2, rank_matrix.T)))
#     result = 12 * n / (k * (k + 1)) * (sumr - k * (k + 1) ** 2 / 4)
#     result = (n - 1) * result / (n * (k - 1) - result)
#     return result
def friedman(n,k,data_matrix): #n:数据集 k：算法个数,data是csv格式，n行k列
    hang, lie = data_matrix.shape
    print(hang)
    print(lie)
    # print(data_matrix)
    data_matrix = data_matrix.values
    XuZhi_mean = list()
    for i in range(lie):
        # print(data_matrix[:,i])
        XuZhi_mean.append(data_matrix[:,i].mean())
    print(XuZhi_mean) #这里输出平均序值
    sum_mean = np.array(XuZhi_mean)

    sum_ri2_mean = (sum_mean ** 2).sum()

    result_Tx2 = (12 * n) * (sum_ri2_mean - ((k * (k + 1) ** 2) / 4))/ (k * (k + 1))
    result_Tf = (n - 1) * result_Tx2 / (n * (k - 1) - result_Tx2)
    return result_Tf
data= pd.read_excel(r"F:\PycharmProjects\ml\EASE\EASE-main\tools\F1.xlsx", index_col=0)
# data = np.asarray(df)
#用法
result = friedman(43,14,data)
print(result)