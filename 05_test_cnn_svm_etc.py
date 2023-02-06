##山ノ井研究室が作成したデータがクラス分類可能か見てみる

import pandas as pd 
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os 
import itertools

df = pd.read_csv("",header=None)
# print(df)
#      0     1     2      3      4      5       6       7       8       9       10      11  ...     73     74     75     76     77     78     79     80     81     82     83     84
# 0     9 -0.84  0.53   3.59  -6.03 -11.37   -7.63   -0.84  -31.37   -3.44   -2.29   -0.61  ...   1.22   6.56   3.21   9.85   5.42   4.50   4.27 -12.82   8.09   2.06  -1.98  16.79
# ..   ..   ...   ...    ...    ...    ...     ...     ...     ...     ...     ...     ...  ...    ...    ...    ...    ...    ...    ...    ...    ...    ...    ...    ...    ...
# 299   7  9.54 -1.83  -1.15  -0.99   9.85   -3.89   -9.54   -5.50    8.85    6.72    6.03  ...  -8.32  -0.53   7.40  -5.42  25.11  25.95  26.41  38.17   1.60   3.51   1.83   6.34
# [300 rows x 85 columns]

recall_ans_df = df.iloc[:,0]#想起画像の列のみを抽出
ch_df = df.iloc[:,1:]
# print(len(recall_ans_df))
# print(recall_ans_df)
# 0      9
#       ..
# 299    7
# Name: 0, Length: 300, dtype: int64
# print(ch_df)
#        1     2      3      4      5       6       7       8       9       10      11      12  ...     73     74     75     76     77     78     79     80     81     82     83     84
# 0   -0.84  0.53   3.59  -6.03 -11.37   -7.63   -0.84  -31.37   -3.44   -2.29   -0.61    3.82  ...   1.22   6.56   3.21   9.85   5.42   4.50   4.27 -12.82   8.09   2.06  -1.98  16.79     
# ..    ...   ...    ...    ...    ...     ...     ...     ...     ...     ...     ...     ...  ...    ...    ...    ...    ...    ...    ...    ...    ...    ...    ...    ...    ...     
# 299  9.54 -1.83  -1.15  -0.99   9.85   -3.89   -9.54   -5.50    8.85    6.72    6.03    4.43  ...  -8.32  -0.53   7.40  -5.42  25.11  25.95  26.41  38.17   1.60   3.51   1.83   6.34     
# [300 rows x 84 columns]

##同クラス内の係数の相関を調べる
call_pic_list = range(1,11)
# os.makedirs("out/corr/same_class/cmp_ch",exist_ok=True)
# for call_pic_temp in call_pic_list:#チャンネル間の比較は意味ない
#     plt.clf()
#     same_class_df = pd.DataFrame()
#     same_class_df = df[df[0] ==call_pic_temp]
#     same_class_df_corr = same_class_df.corr().reset_index()
#     sns.heatmap(same_class_df_corr,vmax=1,vmin=-1,center=0)
#     plt.savefig("out/corr/same_class/cmp_ch/heat_map_call_pic_compare_ch" +str(call_pic_temp) + ".png")

os.makedirs("out/corr/same_class/cmp_try",exist_ok=True)

count = 0
for call_pic_temp in call_pic_list:
    plt.clf()
    count +=1
    same_class_df = pd.DataFrame()
    same_class_df = df[df[0] ==call_pic_temp].reset_index(drop=True)
    same_class_df = same_class_df.T

    same_class_df_corr = same_class_df.corr().reset_index(drop=True)
    sns.heatmap(same_class_df_corr,vmax=1,vmin=-1,center=0)
    plt.savefig("out/corr/same_class/cmp_try/heat_map_call_pic_compare_try_pic_num" +str(call_pic_temp) + ".png")


##違うクラスの共分散を比較する
#2kuras
for pair in itertools.combinations(call_pic_list,2):
    (right , left )= pair




print("すべての作業が実行されました　")