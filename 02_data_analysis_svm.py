import pandas as pd 
##データの読みだし
test_csv_path = r""
df_path = test_csv_path
df = pd.read_csv(df_path)
#print(df.columns)
# ['TIME', ' Fp1-REF', ' Fp2-REF', ' F3-REF', ' F4-REF', ' C3-REF', ' C4-REF', ' P3-REF', ' P4-REF', ' O1-REF', ' O2-REF', ' F7-REF',
# ' F8-REF', ' T3-REF', ' T4-REF', ' T5-REF', ' T6-REF', ' Fz-REF',
# ' Cz-REF', ' Pz-REF', ' 30-31', ' A1-REF', ' A2-REF', ' EXT',
# 'recall_pic_num', 'recall_total']


## 指定した画像に対してのデータを構成する
sampling_rate_ms = 25  #サンプリングレート
sampling_start_posision_ms = 400 #サンプリングの開始位置
sampling_end_posision_ms = 900
delta = 1 #いわゆるデータのずれ
multiple_mode = 1  #1想起から得られるデータを何倍するか論文では　--y_pred_mode という表現をされている
pic_num = 0 #ほしい画像のデータの番号
data_label : int  = 0 #　通常のサンプリング，±1サンプリングをそれぞれ区別する 
out_sampled_df = pd.DataFrame() #分析部分に出力するdf
# try_total = 0 #特定画像に画像に対する想起回数の合計　平均して10回になるはず（ランダムに画像を想起させているから）

#特定の画像に対するデータをつくる
for pic_num in range(1,11):
    special_pic_temp_df = df[df["recall_pic_num"]==pic_num] #特定の画像想起だけのデータを作成
    for temp_recall_total  in special_pic_temp_df["recall_total"].unique():#　特定の画像想起のうち1想起を取り出す
        #print("temprevall total " + str(temp_recall_total))
        special_recall_df = special_pic_temp_df[special_pic_temp_df["recall_total"] == temp_recall_total]#特定の画像だけのdfからさらに1想起だけのdfを作る

        temp_df = special_recall_df.iloc[sampling_start_posision_ms:sampling_end_posision_ms+1:sampling_rate_ms]#start end どちらも取得するため　+1 とする
        #data_labelの列を作成し，結合する
        # temp_df_1["data_label"]= data_label#この形でやるとエラーをはく
        label_list = []
        for index in range(len(temp_df)):
            label_list.append(data_label)
        label_df = pd.DataFrame({"data_label" :label_list})
        temp_df.reset_index(drop=True,inplace=True)#スライスして作成したものを結合するときはindex reset
        temp_df = pd.concat([temp_df,label_df],axis = 1)#横に結合
        data_label +=1 #

        #ｘ倍モードの実装
        if(multiple_mode %2 !=0 ):#モードが奇数の時
            if(multiple_mode ==1 ):#1倍モードであるならデータを増やさなくてよい
                pass
            else:#x倍モードであるとき
                total_multiple = multiple_mode //2 #3倍モードなら 前後1本，
                temp_multiple = 1
                while temp_multiple <= total_multiple:
                    #print("x倍のモードを実行中です, temp_multiple:"+str(temp_multiple))
                    temp_df_1 = special_recall_df.iloc[sampling_start_posision_ms - delta*temp_multiple : sampling_end_posision_ms+1-delta*temp_multiple : sampling_rate_ms]## 3倍モードの2本目
                    label_list = []
                    for index in range(len(temp_df_1)):
                        label_list.append(data_label)
                    label_df = pd.DataFrame({"data_label" :label_list})
                    temp_df_1.reset_index(drop=True,inplace=True)#スライスして作成したものを結合するときはindex reset
                    temp_df_1 = pd.concat([temp_df_1,label_df],axis=1)#横に結合
                    data_label +=1
        
                    temp_df_2 = special_recall_df.iloc[sampling_start_posision_ms + delta*temp_multiple : sampling_end_posision_ms+1+delta*temp_multiple : sampling_rate_ms]#3倍モードの3本目
                    label_list = []
                    for index in range(len(temp_df_2)):
                        label_list.append(data_label)
                    label_df = pd.DataFrame({"data_label" :label_list})
                    temp_df_2.reset_index(drop=True,inplace=True)#スライスして作成したものを結合するときはindex reset
                    temp_df_2 = pd.concat([temp_df_2,label_df],axis=1)#横に結合
                    data_label +=1
                    
                    temp_df.reset_index(drop=True,inplace=True)#スライスして作成したものを結合するときはindex reset
                    temp_df_1.reset_index(drop=True,inplace=True)
                    temp_df_2.reset_index(drop=True,inplace=True)
                    temp_df = pd.concat([temp_df,temp_df_1,temp_df_2],axis=0)

                    temp_multiple +=1 
        else:#モードが偶数
            print("multiple_mode は奇数の必要があります")


        # print("data length per 1 ch : " + str(len(temp_df_1)))
        out_sampled_df = pd.concat([out_sampled_df,temp_df],axis=0)#縦に結合
        # print("1画像について複数回の想起によって得られるデータ本数:" + str(len(out_sampled_df["data_label"].unique())))
    
out_sampled_df.reset_index(drop= True,inplace=True)
# print(out_sampled_df)# recall_pic_num 順だから，recall_totall が 1-0 の範囲で始まる
#                TIME   Fp1-REF   Fp2-REF   F3-REF   F4-REF   C3-REF   C4-REF   P3-REF   P4-REF   O1-REF   O2-REF  ...   T6-REF   Fz-REF   Cz-REF   Pz-REF   30-31   A1-REF   A2-REF   EXT  recall_pic_num  recall_total  data_label
# 0     000:00:56.542      4.89     -2.82     5.34     0.92     2.67    -1.76     2.44    -0.46   -10.46    -8.78  ...    -6.87     4.05     0.00     2.67   -3.51    -5.34   -10.23     2               1             6           0
# 1     000:00:56.567      0.15     -7.25     0.00     1.83    -2.52    -4.35    -1.60    -9.47     4.20   -10.92  ...    -8.09     0.53    -3.05    -1.83  -11.45    -1.60    -7.02    -8               1             6           0
# 2     000:00:56.592     -1.45      5.80    -2.06    -1.45    -3.66     1.15    -0.38     2.14    -4.12   -16.79  ...     1.91    -3.89    -0.69    -0.84  -12.37    -8.24    -3.21    -9               1             6           0 
# ...             ...       ...       ...      ...      ...      ...      ...      ...      ...      ...      ...  ...      ...      ...      ...      ...     ...      ...      ...   ...             ...           ...         ... 
# 6298  000:16:30.842    -18.85    -19.85   -27.71   -32.52   -37.40   -38.02   -46.11   -47.63   -35.88   -43.59  ...   -41.37   -33.36   -43.59   -46.26  -24.43   -30.08   -31.45     1              10            98         299 
# 6299  000:16:30.867    -17.02    -14.81   -23.89   -19.47   -27.86   -16.64   -36.03   -29.54   -39.77   -14.27  ...   -28.09   -20.38   -25.65   -30.92  -20.46   -26.56   -24.58    -5              10            98         299 

print("データの作成完了")

##データのチャンネル数を減らしていく
ch_need_list = ["recall_pic_num","recall_total","data_label"]
data_ch_list = [" Fp2-REF"," F4-REF"," C3-REF"," F8-REF"]
# data_ch_list = [' Fp1-REF', ' Fp2-REF', ' F3-REF', ' F4-REF', ' C3-REF', ' C4-REF', ' P3-REF', 
#                       ' P4-REF', ' O1-REF', ' O2-REF', ' F7-REF',' F8-REF', ' T3-REF', ' T4-REF',
#                        ' T5-REF', ' T6-REF', ' Fz-REF',' Cz-REF', ' Pz-REF', ' 30-31', ' A1-REF', ' A2-REF']

data_col_list = data_ch_list + ch_need_list
#print(data_col_list)
#[' Fp2-REF', ' F4-REF', ' C3-REF', ' F8-REF', 'recall_pic_num','recall_total', 'data_label']
out_sampled_df = out_sampled_df[data_col_list] #データ分析に用いるデータ


##作成したデータに対してジャックナイフ方を用いた正準判別分析を適応する
from statistics import multimode
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(n_components=2,store_covariance=True)


##正準判別分析
#https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

y_ans  = []
y_pred = []
for A_recall_num in out_sampled_df["recall_total"].unique():#1回の想起だけを取り出す
    # print(A_recall_num)

    #ジャックナイフ法の適用 (特定の想起を除外する)
    A_recall_df = out_sampled_df[out_sampled_df["recall_total"] == A_recall_num]#特定の想起
    train_df = out_sampled_df[out_sampled_df["recall_total"] != A_recall_num]#それ以外のデータ

    #学習#
    X_train = []
    Y_train = []
    for temp_data_label in train_df["data_label"].unique():#dfからリスト形式に変換trainだけ
        # print(temp_data_label)
        temp_df = train_df[train_df["data_label"] == temp_data_label]
        temp_train_x_list = temp_df[data_ch_list].to_numpy().flatten().tolist()##特定のチャンネルのデータだけにし，特徴量リストを作る
        temp_train_y_int = temp_df["recall_pic_num"].iat[1]#正解ラベルを一つ入力する[2,2,2] → 2だけ使いたい
        
        X_train.append(temp_train_x_list)
        Y_train.append(temp_train_y_int)
        #print(temp_y_int)

    clf.fit(X_train,Y_train)

    #予測#
    X_test = []
    A_recall_pred = []
    for temp_data_label in A_recall_df["data_label"].unique():#dfからリスト形式に変換trainだけ
        # print(temp_data_label)
        temp_df = pd.DataFrame()
        temp_df = A_recall_df[A_recall_df["data_label"] == temp_data_label]
        temp_test_x_list = temp_df[data_ch_list].to_numpy().flatten().tolist()##特定のチャンネルのデータだけにし，特徴量リストを作る

        X_test.append(temp_test_x_list)

    A_recall_pred = clf.predict(X_test)
    print(A_recall_pred)

    
    #最頻値を1回の想起の解答とする.
    # 例：A_recall_pred = [1,2,2] なら　2を画像想起の予測とする
    y_pred_mode = multimode(A_recall_pred)#最頻値をリストとして返す
    if len(y_pred_mode) >2 : # 最頻値が複数の場合
        y_pred_mode_int = 100
    else:
        y_pred_mode_int = y_pred_mode[0]

    y_ans.append(A_recall_df["recall_pic_num"].iat[0])
    y_pred.append(y_pred_mode_int)





##実際の解答と予測をチェックする
print(y_ans)
print(y_pred)

##正答率を出す
score = 0
correct = 0
for index in range(len(y_ans)): #正解数を数える
    if y_ans[index] == y_pred[index]:
        correct +=1

score = float(correct/len(y_pred)) *100.0#正答率を百分率で出力する
print("score:" + str(score) +"%")

