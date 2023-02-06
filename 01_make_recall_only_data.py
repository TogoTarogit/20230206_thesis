import numpy as np
import pandas as pd 
import os 

##元となるデータ
eeg_data_csv_dir = r""
pic_order_info_txt_dir = r""

##中間データ
eeg_df = pd.read_csv(eeg_data_csv_dir)
trigger_df = pd.DataFrame()##トリガーの情報
pic_order_list =[]
##出力データ
recall_only_df = pd.DataFrame()

print("入力データはeeg:" + str(eeg_data_csv_dir) +"写真整列順：" + str(pic_order_info_txt_dir))

####
#ここからが処理の内容
####

##トリガーに関するdf を作っていく
def make_trigger_data (eeg_data_csv_dir = "" ):
    eeg_df = pd.read_csv(eeg_data_csv_dir) 
    max_data = eeg_df[" EXT"].max()
    min_data = eeg_df[" EXT"].min()
    threshold = (max_data + min_data )/2

    ##トリガーの情報を作っていく(トリガーはインパルスだけど実際は幅を持っている)
    trigger_tapple_index_list = [] # (begging_triiger , end_trigger_index )
    bool_in_trigger = False #トリガーの”中”にいるかの判別
    begin_trigger_index = 0
    end_trigger_index = 0
    
    # EXTの出力に対して　ある一定の値より小さいならそれをトリガーとして疑う
        # 前１つと現在と後ろ２つの計４要素の平均と閾値との比較を行い，トリガーかどうかを判定する
        # トリガー内と判定したら，trigger_in_bool をTrueとする
        # トリガー脱出は同じ要領で判定を行う．
        # 出力はタプルのリスト型として出力する
    for index ,ext_temp in eeg_df[" EXT"].items():#EXTの値とindex を取り出す
        #print(type(ext_temp))  # int 
        #print(indx)  #int
        #print(eeg_df[" EXT"][index])
        if bool_in_trigger == False and ext_temp < threshold:#find maybe trigger #トリガー内ではなく　現在のext がthreshold より小さい→多分トリガー
            average_around_ext_temp = (eeg_df[" EXT"][index-2] +eeg_df[" EXT"][index-1] + eeg_df[" EXT"][index] + eeg_df[" EXT"][index+1] +eeg_df[" EXT"][index+2]) /5
            if  average_around_ext_temp < threshold: 
                #print("now in trigger, index is ", str(index))
                bool_in_trigger = True #  NOW in trigger 
                begin_trigger_index = index
                #print(bool_in_trigger,begin_trigger_index,end_trigger_index)

        if bool_in_trigger == True and ext_temp >= threshold:# in trigger ,however tempExt is higher than  threshold
            #print("now , in trigger and ext_temp higher than threshold  ")
            average_around_ext_temp = (eeg_df[" EXT"][index-2] +eeg_df[" EXT"][index-1] + eeg_df[" EXT"][index] + eeg_df[" EXT"][index+1] +eeg_df[" EXT"][index+2]) /5
            if  average_around_ext_temp > threshold : 
                #print("now out of trigger, index is ", str(index))
                bool_in_trigger = False #  NOW out of  trigger 
                end_trigger_index = index
                tup = begin_trigger_index,end_trigger_index
                trigger_tapple_index_list.append(tup) 
                #print(bool_in_trigger,begin_trigger_index,end_trigger_index)

    #トリガーの数を数えるトリガーの数は　想起回数x2 となり50の倍数となる
    if (len(trigger_tapple_index_list)% 50 != 0):
        print("trigger の計算に失敗している可能性があります")
        print(trigger_tapple_index_list)
        print(len(trigger_tapple_index_list))

    #トリガーの発出経過時間を計算する（一部長かったりすると，トリガーの検出に失敗している）
    for tapple in trigger_tapple_index_list:
        beggin , end = tapple
        diff = end - beggin
        #print(beggin, end ,diff)
        if (diff > 1000):
            print("トリガーの発出時間が不正です diff : " + str(diff))
    #print(type(trigger_tapple_index_list)) 

    #タプルをpandas df に変換する
    list_col = ["trigger_start_index","trigger_end_index"]
    triggers_begin_and_end_df = pd.DataFrame(columns=list_col)
    #print(triggers_begin_and_end_df)
    #tapple to pandas df 
    for temp_tapple in trigger_tapple_index_list:
        beggin, end = temp_tapple
        temp_trigger = pd.DataFrame({"trigger_start_index":[beggin],"trigger_end_index":[end]})
        triggers_begin_and_end_df = pd.concat([triggers_begin_and_end_df,temp_trigger],ignore_index=True)

    #データのチェックと中間データを保存する
    #triggers_begin_and_end_df = triggers_begin_and_end_df.rename(columns= {'0':'start' ,'1':'end'})

    if(len(triggers_begin_and_end_df)%50 !=0):#200 :10picture *(10times ) *(recall and show =2)
        print("ALERT, some trigger is broken, trigger count is :" + str(len(triggers_begin_and_end_df)))
    os.makedirs("./progress" ,exist_ok=True)
    os.makedirs("./progress/trigger_raw_data" ,exist_ok=True)


    triggers_begin_and_end_df.to_csv("./progress/" +"trigger_" + eeg_data_csv_dir)
    #print(triggers_begin_and_end_df.columns)
    #print(triggers_begin_and_end_df)
    # ext = eeg_df[" EXT"]
    # datasize= 98000
    # ext = ext[97000:datasize]
    # plt.clf()
    # ext.plot()
    # plt.savefig("test_2")
    return triggers_begin_and_end_df
trigger_df = make_trigger_data(eeg_data_csv_dir)
# print(trigger_df)
#     trigger_start_index trigger_end_index
# 0                   743              1048
# 1                  5749              6076

##画像出力順を抽出する
def make_pic_order_int_list(path_str):
    #path
    test_txt_path = path_str
    f = open(test_txt_path,'r')
    text_list = f.readlines()
    #print(text_list)
    pic_order_line_list = []
    for index in range(len(text_list)):
        #print(index)
        if index%2 ==0:#偶数行が想起画像のデータ
            pic_order_line_list.append(text_list[index])
            #print(pic_order_line_list)
        else:
            pass

    #データ整形
    #各データに対して09 とか画像識別だけのデータにする
    for index in range(len(pic_order_line_list)):
        #print(index)
        pic_order_line_list[index] =int( pic_order_line_list[index][0:2])# 先頭の２文字を抽出し，int に変換しこれを代入する

    #print(pic_order_line_list)
    #print(len(pic_order_line_list))
    length = len(pic_order_line_list)
    if(length%50 ==0):##画像出力順のテスト
        pass
    else:
        print("ERROR!! pic_order_list_lenght is not 100 , this length is :" + str(length))
    pic_order_int_list = pic_order_line_list
    
    #中間データとして保存
    os.makedirs("./progress/pic_order_raw_data" ,exist_ok=True)
    df = pd.DataFrame(pic_order_int_list)
    df.to_csv("./progress/" +"pic_order_" + path_str[:-2] +".csv")
    return pic_order_int_list
pic_order_list = make_pic_order_int_list(pic_order_info_txt_dir)
# print(pic_order_list)
#[9, 5, 8, 2, 7, 1, 6, 10, 3, 4, 10, 7, 8, 2, ... 2, 1, 10, 8, 7]

##画像想起中のみのデータにする
def make_recall_only_df (eef_df_1,trigger_df_1,pic_order_list_1):
    eef_df = eef_df_1
    trigger_df = trigger_df_1
    pic_order_list = pic_order_list_1

    recall_only_df = pd.DataFrame()
    recall_data_length_ms = 2000
    out_path = "./out/recall_only_" + eeg_data_csv_dir

    #出力データの初期化
    os.makedirs("./out" ,exist_ok=True)
    os.makedirs("./out/recall_only_raw_data/" ,exist_ok=True)

    f = open(out_path,"w")
    f.close()
    recall_total = 0 #総想起回数
    for trigger_index in range(len(trigger_df)):
        recall_total = (trigger_index // 2) +1 ##index を2で割った商+1が想起回数　例：index=1 のトリガーの後に想起が来る　これは
        
        if(trigger_index %2 ==1):#インデックスの奇数回目のトリガーの後の2000msがデータとして必要
            #print("this is recall trigger index is :", str(trigger_num) )
            temp_recall_pic_num = pic_order_list[recall_total-1]#1回目の想起の画像はrecall_num_total-1 のindex 
            
            #想起1回におけるEEGデータを抽出する
            temp_data_start_point = trigger_df["trigger_end_index"][trigger_index]
            temp_data_end_point = temp_data_start_point + recall_data_length_ms
            temp_df = eeg_df[temp_data_start_point:temp_data_end_point].copy()#copy を外すとchain indexing というエラーが生じる
            #print(temp_recall_total,temp_recall_pic_num,temp_data_start_point)

            #1回の想起データに累計想起回数と想起画像のラベルを付加する
            temp_df["recall_pic_num"] = temp_recall_pic_num
            temp_df["recall_total"] = recall_total
            #print(temp_df.columns)
            # print(temp_df[0:5])
            
            #想起ごとにdf に保存していく
            recall_only_df = pd.concat([recall_only_df,temp_df],axis=0)#列名が同じものを結合する
    if(recall_total %50 != 0):
        print("総想起回数が不正です ,recall_total :" +str(recall_total))
    #最後にcsvに保存する
    recall_only_df.to_csv(out_path,mode=r'a',index=False,encoding="utf-8-sig") 
    return recall_only_df
recall_only_df = make_recall_only_df(eeg_df,trigger_df,pic_order_list)
# print(recall_only_df)
#                   TIME   Fp1-REF   Fp2-REF   F3-REF   F4-REF   C3-REF   C4-REF   P3-REF  ...   Cz-REF   Pz-REF   30-31   A1-REF   A2-REF   EXT  recall_pic_num  recall_total
# 6076     000:00:06.076      9.85      9.47    19.24    16.34    23.89    21.76    20.76  ...    15.73    18.93   -5.34     7.56    10.76  -414               9             1          
# 6080     000:00:06.080      6.26      7.25    11.22    16.11    16.79    18.17    11.45  ...    10.31     9.54   -0.53     2.06     2.60  -369               9             1        
# ...                ...       ...       ...      ...      ...      ...      ...      ...  ...      ...      ...     ...      ...      ...   ...             ...           ...              
# 1011991  000:16:51.991    120.99     94.27   148.78   152.82   162.29   167.25   182.21  ...   168.40   181.83  292.14   173.82   182.60     0               7           100        
# [200000 rows x 26 columns]

print("実行完了")


# print(eeg_df)    

