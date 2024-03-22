import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp # https://pypi.org/project/scikit-posthocs/
import stac
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]   # 设置显示中文字体

df = pd.read_excel("c.xlsx", index_col=0)
data = np.asarray(df)

print("Read data")
num_datasets, num_methods = data.shape
print("Methods:", num_methods, "Datasets:", num_datasets)

alpha = 0.05 # Set this to the desired alpha/signifance level
stat, p, ranking, rank_cmp=stac.friedman_test(*np.transpose(data))
# stat, p = stats.friedmanchisquare(*data)
print(stat)
print(p)
reject = p <= alpha
print(reject)
if not reject:
    raise Exception("Exiting early. The rankings are only relevant if there was a difference in the means i.e. if we rejected h0 above")
# Helper functions for performing the statistical tests
def generate_scores(method, method_args, data, labels):
    pairwise_scores = method(data, **method_args) # Matrix for all pairwise comaprisons
    pairwise_scores.set_axis(labels, axis='columns', inplace=True) # Label the cols
    pairwise_scores.set_axis(labels, axis='rows', inplace=True) # Label the rows, note: same label as pairwise combinations
    return pairwise_scores

def plot(scores):
    # Pretty plot of significance
    cmap = ['1', '#fb6a4a', '#08306b', '#4292c6', '#c6dbef']
    heatmap_args = {'cmap': cmap,'linewidths': 0.5, 'linecolor': 'white', 'square': True,
                    'cbar_ax_bbox': [0.78, 0.38, 0.04, 0.3]}
    fix, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('G-mean指标Nemenyi后续检验')
    sp.sign_plot(scores, **heatmap_args)
nemenyi_scores = generate_scores(sp.posthoc_nemenyi_friedman, {}, data, df.columns)
print(nemenyi_scores)
plot(nemenyi_scores)
plt.show()

statistic, p_value, ranking, rank_cmp  = stac.friedman_test(*np.transpose(data))
ranks = {key: rank_cmp[i] for i, key in enumerate(list(df.columns))}
print(ranks)
comparisons, z_values, p_values, adj_p_values = stac.holm_test(ranks, control="Proposed Method")
adj_p_values = np.asarray(adj_p_values)
for method, rank in ranks.items():
    print(method +":", "%.2f" % rank)
holm_scores = pd.DataFrame({"p": adj_p_values, "sig": adj_p_values < alpha}, index=comparisons)
print(holm_scores)

'''
#画图
F1=[2.95 ,6.86,	5.58 ,8.35 ,	10.21 ,9.21 ,6.23 ,10.21 ,8.12,12.93 ,10.72 ,7.42 ,6.51 ,6.93 ,7.33]
MCC=[2.91,6.09 ,	5.23 ,7.93 ,9.51 ,8.26 ,	7.26 ,9.51 ,7.51 ,13.44 ,11.67 ,8.84 ,7.95 ,	5.56 ,7.88 ]
AUC_PR=[2.88 ,3.44 	,5.79 	,5.23 ,	6.21, 	8.07 	,5.40 ,	7.51 ,	6.44 	,14.40 ,	13.56,12.12,	11.63 ,4.44 ,11.60 ]
G_mean=[5.47 ,	5.23 	,6.23 ,	12.88, 	5.53 ,	4.02 	,9.53 ,	5.67,5.60 ,	11.70 	,10.88 ,	11.33 ,	11.81 ,	2.19 ,11.47]
# F1=[3.31,5.59,5.02,7.28 ,9.57 ,8.50 ,5.86 ,8.48 ,6.91 ,11.83 ,10.34 ,6.22 ,6.97 ,6.36]
# MMC=[3.34 ,5.16 ,4.78 ,6.88 ,9.12 ,7.81 ,6.76 ,8.02 ,6.41 ,12.29 ,11.03 ,5.29 ,7.93 ,7.41 ]
# AUC_PR=[2.97 ,3.16,	5.55 ,5.02 ,6.02 ,7.62 ,5.21 ,6.50 ,6.00 ,13.12 ,12.81 ,4.97 ,11.17 ,10.83 ]
# G_mean=[5.43 ,5.00 ,6.03 ,11.22 ,5.95 ,4.17 ,8.47 ,5.00 ,5.26 ,11.34 ,10.84 ,2.29 ,10.38 ,10.84 ]
F1.reverse()
MCC.reverse()
AUC_PR.reverse()
G_mean.reverse()
plt.rc('font',family='Times New Roman')
y=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
CD=3.26
h_CD=CD/2
fig=plt.figure(figsize=(22,14),dpi=160)

ax1=fig.add_subplot(2,2,1)
_alg_=F1
ax1.scatter(_alg_,y,s=100,c='black')
for i in range(len(y)):
    yy=[y[i],y[i]]
    xx=[_alg_[i]-h_CD,_alg_[i]+h_CD]
    ax1.plot(xx, yy,linewidth=3.0)
    # plt.plot([_alg_[i] + h_CD, _alg_[i] + h_CD], [1, y[i]], linestyle='--')
    ax1.set_yticks(range(16,0,-1),labels=['','TSSM-BIM',	'EASE',	'SPE',	'SBag',	'RUSBoost',	'UBag',	'SBoost',	'BC',	'ECUBoost',	'AdaOBU',	'BoostOBU',
                                     	'AS',	'BS','HUE','SH'],size=20)
# ax1.plot([_alg_[-1]+h_CD, _alg_[-1]+h_CD],[1,y[-1]],linestyle='--',color='r')#竖线
ax1.set_xticks(range(0,16,1),labels=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'],size=20)
ax1.set_xlabel("F1 Average Order",size=20)
# plt.title("F1次序图",size=20)
# plt.savefig("title"+'.png',format='PNG',dpi=500,bbox_inches='tight', pad_inches = +0.1)

ax1=fig.add_subplot(2,2,2)
_alg_=MCC
ax1.scatter(_alg_,y,s=100,c='black')
for i in range(len(y)):
    yy=[y[i],y[i]]
    xx=[_alg_[i]-h_CD,_alg_[i]+h_CD]
    ax1.plot(xx, yy,linewidth=3.0)
    # plt.plot([_alg_[i] + h_CD, _alg_[i] + h_CD], [1, y[i]], linestyle='--')
    ax1.set_yticks(range(16,0,-1),labels=['','TSSM-BIM',	'EASE',	'SPE',	'SBag',	'RUSBoost',	'UBag',	'SBoost',	'BC',	'ECUBoost',	'AdaOBU',	'BoostOBU',
                                     'AS',	'BS','HUE'	,'SH'],size=20)
# ax1.plot([_alg_[-1]+h_CD, _alg_[-1]+h_CD],[1,y[-1]],linestyle='--',color='r')
ax1.set_xticks(range(0,16,1),labels=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'],size=20)
ax1.set_xlabel("MCC Average Order",size=20)

ax1=fig.add_subplot(2,2,3)
_alg_=AUC_PR
ax1.scatter(_alg_,y,s=100,c='black')
for i in range(len(y)):
    yy=[y[i],y[i]]
    xx=[_alg_[i]-h_CD,_alg_[i]+h_CD]
    ax1.plot(xx, yy,linewidth=3.0)
    # plt.plot([_alg_[i] + h_CD, _alg_[i] + h_CD], [1, y[i]], linestyle='--')
    ax1.set_yticks(range(16,0,-1),labels=['','TSSM-BIM',	'EASE',	'SPE',	'SBag',	'RUSBoost',	'UBag',	'SBoost',	'BC',	'ECUBoost',	'AdaOBU',	'BoostOBU',
                                     'AS',	'BS','HUE','SH'],size=20)
# ax1.plot([_alg_[-1]+h_CD, _alg_[-1]+h_CD],[1,y[-1]],linestyle='--',color='r')
ax1.set_xticks(range(0,16,1),labels=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'],size=20)
ax1.set_xlabel("AUC-PR Average Order",size=20)

ax1=fig.add_subplot(2,2,4)
_alg_=G_mean
ax1.scatter(_alg_,y,s=100,c='black')
for i in range(len(y)):
    yy=[y[i],y[i]]
    xx=[_alg_[i]-h_CD,_alg_[i]+h_CD]
    ax1.plot(xx, yy,linewidth=3.0)
    # plt.plot([_alg_[i] + h_CD, _alg_[i] + h_CD], [1, y[i]], linestyle='--')
    ax1.set_yticks(range(16,0,-1),labels=['','TSSM-BIM',	'EASE',	'SPE',	'SBag',	'RUSBoost',	'UBag',	'SBoost',	'BC',	'ECUBoost',	'AdaOBU',	'BoostOBU',
                                      'AS',	'BS','HUE','SH'],size=20)
# ax1.plot([_alg_[-1]+h_CD, _alg_[-1]+h_CD],[1,y[-1]],linestyle='--',color='r')
ax1.set_xticks(range(0,16,1),labels=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'],size=20)
ax1.set_xlabel("G-mean Average Order",size=20)



plt.savefig("title"+'.png',format='PNG',dpi=600,bbox_inches='tight', pad_inches = +0.1)

plt.show()

'''