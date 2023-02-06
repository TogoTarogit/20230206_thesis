import pandas as pd
import numpy as np

###山ノ井研究室の処理により得られた学習データが再現可能か調べる
##本来は
## 入力データ
yama_df = pd.DataFrame()
taro_df = pd.DataFrame()
def input_data_train_compare(yama_df,taro_df):

    yama_test_data_df = pd.read_csv("")
    yama_df = yama_test_data_df
    # print(yama_test_data_df)
    #      9   -.84    .53   3.59  -6.03  -11.37   -7.63  -.84.1  -31.37   -3.44   -2.29  ...    6.56    3.21   9.85   5.42  4.5.1    4.27  -12.82   8.09   2.06   -1.98  16.79
    # 0    5   2.90   2.90   7.94  12.67  -88.70 -136.18 -157.48 -104.89 -196.18 -278.17  ...   23.89   28.70  35.11  11.60  23.82   24.73   20.23   9.31   5.04    1.91   1.98
    # 1    8  -8.40 -14.58 -23.97  -4.12    5.95   -7.10  -16.03   -5.04  -12.37  -10.38  ...  -12.98  -14.73 -18.17  -5.57  -8.85  -16.56   -7.94  -0.46  -1.91   -5.88 -10.46
    # ..  ..    ...    ...    ...    ...     ...     ...     ...     ...     ...     ...  ...     ...     ...    ...    ...    ...     ...     ...    ...    ...     ...    ...
    # 97   8   5.57   8.55   0.92  -8.93   20.61   21.37   16.56   40.61    4.43    4.27  ...   -9.01  -10.46 -10.99  -7.79  -1.15    0.84  -22.82 -12.60  -6.11   -8.47 -21.60       
    # 98   7   7.40  -5.19  -3.59  -8.85   10.99   -1.76   -8.32    1.91    1.45    3.66  ...    2.37   11.76  -9.24  17.02  19.85   19.69   28.63  -0.15   2.98    0.53   3.82       
    # [99 rows x 85 columns]

    out_sampled_df = pd.DataFrame() #分析部分に出力するdf
    ##データの読みだし
    df_path = r""
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
    multiple_mode = 101  #1想起から得られるデータを何倍するか論文では　--y_pred_mode という表現をされている
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
    #print(out_sampled_df)
    # 0     000:00:56.542      4.89     -2.82     5.34     0.92     2.67    -1.76     2.44  ...     2.67   -3.51    -5.34   -10.23     2               1             6           0    
    # 1     000:00:56.567      0.15     -7.25     0.00     1.83    -2.52    -4.35    -1.60  ...    -1.83  -11.45    -1.60    -7.02    -8               1             6           0    
    # ...             ...       ...       ...      ...      ...      ...      ...      ...  ...      ...     ...      ...      ...   ...             ...           ...         ...    
    # 6298  000:16:30.842    -18.85    -19.85   -27.71   -32.52   -37.40   -38.02   -46.11  ...   -46.26  -24.43   -30.08   -31.45     1              10            98         299    
    # 6299  000:16:30.867    -17.02    -14.81   -23.89   -19.47   -27.86   -16.64   -36.03  ...   -30.92  -20.46   -26.56   -24.58    -5              10            98         299    
    # [6300 rows x 27 columns]

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

    taro_df = out_sampled_df
    return yama_df,taro_df
   
# yama_df , taro_df=input_data_train_compare(yama_df,taro_df)  

def input_data_recall_compare(yama_df,taro_df):#想起のデータフレームに同じデータ列があるか調べる
    yama_df =  pd.read_csv("")
    taro_df = pd.read_csv(r"")
    return yama_df,taro_df
yama_df , taro_df = input_data_recall_compare(yama_df,taro_df)
print("データの構成が完了") 
# print(yama_df)
# print(taro_df)
#      9   -.84    .53   3.59  -6.03  -11.37   -7.63  -.84.1  -31.37   -3.44   -2.29  ...    6.56    3.21   9.85   5.42  4.5.1    4.27  -12.82   8.09   2.06   -1.98  16.79
# ...
# 98   7   7.40  -5.19  -3.59  -8.85   10.99   -1.76   -8.32    1.91    1.45    3.66  ...    2.37   11.76  -9.24  17.02  19.85   19.69   28.63  -0.15   2.98    0.53   3.82       

# [99 rows x 85 columns]
#                TIME   Fp1-REF   Fp2-REF   F3-REF   F4-REF   C3-REF   C4-REF   P3-REF  ...   Pz-REF   30-31   A1-REF   A2-REF   EXT  recall_pic_num  recall_total  data_label
# 0     000:00:56.542      4.89     -2.82     5.34     0.92     2.67    -1.76     2.44  ...     2.67   -3.51    -5.34   -10.23     2               1             6           0    
# ...
# 6299  000:16:30.867    -17.02    -14.81   -23.89   -19.47   -27.86   -16.64   -36.03  ...   -30.92  -20.46   -26.56   -24.58    -5              10            98         299    
# [6300 rows x 27 columns]

# # 処理
#究室は画像も含めて85列　論文通り4ch 21サンプリングだとする
#研究室　98列目の7.40  -5.19  -3.59　がおなじ行にないかしらべる
# search_value = [7.40 , -5.19 , -3.59,-8.85]
# find_all  value: pic_num,s_index ,index,col: 4,3,78128,21
# search_value = [10.99  , -1.76 ,  -8.32  ,  1.91]
# 該当なし？？？？？
search_value = [ -0.84   , 0.53   ,3.59 , -6.03 ]
# find_all  value: pic_num,s_index ,index,col: 3,3,17072,16
# find_all  value: pic_num,s_index ,index,col: 4,3,104039,21
# find_all  value: pic_num,s_index ,index,col: 3,3,109719,22
# find_all  value: pic_num,s_index ,index,col: 5,3,115939,22
# find_all  value: pic_num,s_index ,index,col: 7,3,122322,13
#該当5つあり

search_index = 0
for index in range(len(taro_df)):
    # print(index)
    for index_col in range(len(taro_df.columns)):
        value = taro_df.iat[index,index_col]
        # print(value)
        if(value == search_value[search_index]):
            search_index +=1
            if search_index == len(search_value)-1:
                print("find_all  value: pic_num,s_index ,index,col: " + str(taro_df.iat[index,len(taro_df.columns)-2]) + ","+str(search_index) + "," +str(index) + "," +str(index_col) )
                search_index=0
                
    search_index  =0 
                 
        
        
        

## 比較結果

print("すべての処理が完了しました")