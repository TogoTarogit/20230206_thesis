import pandas as pd 
##データの読みだし
test_csv_path = r"out\recall_only_YH_2019-12-26.CSV"
df_path = test_csv_path
df = pd.read_csv(df_path)
#print(df.columns)
# ['TIME', ' Fp1-REF', ' Fp2-REF', ' F3-REF', ' F4-REF', ' C3-REF', ' C4-REF', ' P3-REF', ' P4-REF', ' O1-REF', ' O2-REF', ' F7-REF',
# ' F8-REF', ' T3-REF', ' T4-REF', ' T5-REF', ' T6-REF', ' Fz-REF',
# ' Cz-REF', ' Pz-REF', ' 30-31', ' A1-REF', ' A2-REF', ' EXT',
# 'recall_pic_num', 'recall_total']


# 指定した画像に対してのデータを構成する

sampling_rate_ms = 20 #サンプリングレート
sampling_start_posision_ms = 400 #サンプリングの開始位置
sampling_end_posision_ms = 900
delta = 1 #いわゆるデータのずれ
multiple_data = 3 #1想起から得られるデータを何倍するか論文では　--y_pred_mode という表現をされている
pic_num = 0 #ほしい画像のデータの番号
data_label = 0 #　通常のサンプリング，±1サンプリングをそれぞれ区別する 
out_sampled_df = pd.DataFrame() #分析部分に出力するdf
# try_total = 0 #特定画像に画像に対する想起回数の合計　平均して10回になるはず（ランダムに画像を想起させているから）

#特定の画像に対するデータをつくる
for pic_num in range(1,11):
    special_pic_temp_df = df[df["recall_pic_num"]==pic_num] #特定の画像想起だけのデータを作成
    for temp_data_label  in special_pic_temp_df["recall_total"].unique():#　特定の画像の想起
        
        #print("temprevall total " + str(temp_data_label))
        special_recall_df = special_pic_temp_df[special_pic_temp_df["recall_total"] == temp_data_label]#特定の画像だけのdfからさらに1想起だけのdfを作る
        #print(special_recall_df)
        temp_sampling_index = sampling_start_posision_ms
        
        temp_df_1 = special_recall_df.iloc[sampling_start_posision_ms:sampling_end_posision_ms+1:sampling_rate_ms]#start end どちらも取得するため　+1 とする
        temp_df_1.loc[:,"data_label"] = data_label
        data_label +=1
        
        temp_df_2 = special_recall_df.iloc[sampling_start_posision_ms-delta:sampling_end_posision_ms+1-delta:sampling_rate_ms]## 3倍モードの2本目
        temp_df_2.loc[:,"data_label"] = data_label
        data_label +=1
        
        temp_df_3 = special_recall_df.iloc[sampling_start_posision_ms+delta:sampling_end_posision_ms+1+delta:sampling_rate_ms]#3倍モードの3本目
        temp_df_3.loc[:,"data_label"] = data_label
        data_label +=1

        # print("data length per 1 ch : " + str(len(temp_df_1)))
        out_sampled_df = pd.concat([out_sampled_df,temp_df_1,temp_df_2,temp_df_3],axis=0)#縦に結合
        # print("1画像について複数回の想起によって得られるデータ本数:" + str(len(out_sampled_df["data_label"].unique())))

# print(out_sampled_df)# recall_pic_num 順だから，data_labell が 1-0 の範囲で始まる
#                  TIME   Fp1-REF   Fp2-REF   F3-REF   F4-REF   C3-REF   C4-REF   P3-REF   P4-REF  ...   Cz-REF   Pz-REF   30-31   A1-REF   A2-REF   EXT  recall_pic_num  data_label  data_label
# 18400   000:01:31.744     -6.26     -0.76    -3.89    -0.08    -3.59     2.52   -10.84    -3.97  ...     2.44    -6.34  -34.12   -25.95   -20.38    -6               1            10           0
# 18425   000:01:31.769     -2.21      1.68    -5.80     3.28    -6.18     3.82    -8.17    -2.90  ...     0.69    -5.65  -28.02    -5.11    -5.88     5               1            10           0
# ...
# 198876  000:16:54.524    -10.61     -8.55   -12.90   -10.31   -12.75   -12.14   -10.92   -10.61  ...   -12.60    -8.24   -6.11   -12.90    -5.42    -4              10           100         299
# 198901  000:16:54.549    -14.12     -9.08   -14.12   -11.07   -17.56   -13.74   -17.63   -14.96  ...   -14.20   -17.25   -2.75    -9.62    -2.75    -6              10           100         299

# out_sampled_df = out_sampled_df.iloc[::-1]
# print(out_sampled_df)

##データのチャンネル数を減らしていく
ch_need_list = ["recall_pic_num","recall_total","data_label"]


data_ch_list = [" Fp2-REF"," F4-REF"," C3-REF"," F8-REF"]
# data_ch_list = [' Fp1-REF', ' Fp2-REF', ' F3-REF', ' F4-REF', ' C3-REF', ' C4-REF', ' P3-REF', 
#                 ' P4-REF', ' O1-REF', ' O2-REF', ' F7-REF',' F8-REF', ' T3-REF', ' T4-REF',
#                  ' T5-REF', ' T6-REF', ' Fz-REF',' Cz-REF', ' Pz-REF', ' 30-31', ' A1-REF', ' A2-REF']
data_col_list = data_ch_list + ch_need_list
#print(data_col_list)
out_sampled_df = out_sampled_df[data_col_list] #データ分析に用いるデータ
#[' Fp2-REF', ' F4-REF', ' C3-REF', ' F8-REF', 'recall_pic_num','data_label', 'data_label']

from statistics import multimode
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()

##ジャックナイフ法を用いた正準判別分析
#https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

y_ans  = []
y_pred = []
for data_label in out_sampled_df["data_label"].unique():#特定のデータだけを取り出す
    #print(data_label)

    #ジャックナイフ法の適用 (1データを除外する)
    A_data_df = out_sampled_df[out_sampled_df["data_label"] == data_label]#特定の想起
    train_df = out_sampled_df[out_sampled_df["data_label"] != data_label]#それ以外のデータ

    #学習#
    X_train = []
    Y_train = []


    for train_data_label in train_df["data_label"].unique():
        temp_df = pd.DataFrame()
        temp_df = train_df[train_df["data_label"] ==train_data_label]
        temp_train_x_list = temp_df[data_ch_list].to_numpy().flatten().tolist()##特定のチャンネルのデータだけにし，特徴量リストを作る
        temp_train_y_int = temp_df["recall_pic_num"].iat[0]#正解ラベルを一つ入力する[2,2,2] → 2だけ使いたい
        X_train.append(temp_train_x_list)
        Y_train.append(temp_train_y_int)
   
    clf.fit(X_train,Y_train)

    #予測#
    X_test = []
    A_recall_pred = []

    temp_df = pd.DataFrame()
    temp_df = A_data_df
    temp_test_x_list = temp_df[data_ch_list].to_numpy().flatten().tolist()##特定のチャンネルのデータだけにし，特徴量リストを作る

    X_test.append(temp_test_x_list)


    A_recall_pred = clf.predict(X_test)
    print(A_recall_pred)

    #最頻値を1想起の解答とする.
    # 例：A_recall_pred = [1,2,2] なら　2を想起とする
    y_pred_mode = multimode(A_recall_pred)#最頻値をリストとして返す
    if len(y_pred_mode) >2 : # 最頻値が複数の場合
        y_pred_mode_int = 100 
    else:
        y_pred_mode_int = y_pred_mode[0]

    y_ans.append(A_data_df["recall_pic_num"].iat[2])
    y_pred.append(y_pred_mode_int)
    
print(y_ans)
print(y_pred)

##正答率を出す
score = 0
correct = 0
for index in range(len(y_ans)):
    if y_ans[index] == y_pred[index]:
        correct +=1

score = float(correct/len(y_pred)) *100.0

print("score:" + str(score) +"%")

