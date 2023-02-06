import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "MS Gothic"
from scipy import fftpack ,signal
from scipy.fft import fft,ifft, fftfreq
df = pd.read_csv(r"")


data_ch_list = [' Fp1-REF', ' Fp2-REF', ' F3-REF', ' F4-REF', ' C3-REF',
       ' C4-REF', ' P3-REF', ' P4-REF', ' O1-REF', ' O2-REF', ' F7-REF',
       ' F8-REF', ' T3-REF', ' T4-REF', ' T5-REF', ' T6-REF', ' Fz-REF',
       ' Cz-REF', ' Pz-REF', ' EOG', ' A1-REF', ' A2-REF']

need_col_list  = ['TIME', ' EXT', 'recall_pic_num', 'recall_total']
recall_pic_list = range(1,11)
def plot_raw_data (data_frame):
    print("called plot_raw_data")
    df = data_frame
    #特定の画像の想起の各チャンネルについて波形をplotする
    for temp_pic in recall_pic_list:##特定の画像についてloop
        pic_temp_df = df[df["recall_pic_num"]==temp_pic]
        
        for ch_index_int in range(len(data_ch_list)):#チャンネルを選ぶ
            ch_name_str = data_ch_list[ch_index_int]
            ch_df = pd.DataFrame() #行に経過時間，列にtotal_想起
            for recall_total_temp_int in pic_temp_df["recall_total"].unique():#それぞれの想起をloop
                a_try_a_ch_series = pic_temp_df[pic_temp_df['recall_total'] == recall_total_temp_int][ch_name_str]
                a_try_a_ch_series.reset_index(drop = True,inplace = True)
                # a_try_a_ch_series.rename("pic,total_recall:" + str(temp_pic) +"," + str(recall_total_temp_int))

                ch_df = pd.concat([ch_df,a_try_a_ch_series.rename("try_" +str(recall_total_temp_int))],axis= 1)#横に結合

            print("表示処理を開始:(pic,ch) :" + "("+ str(temp_pic) + "," + str(ch_name_str) + ")")
            plt.figure()
            ch_df.plot()
            plt.xlabel("time[ms]")
            plt.ylabel("voltage")
            os.makedirs("out/raw_data_img/",exist_ok=True)
            plt.savefig("out/raw_data_img/" +str(temp_pic) +"_" + ch_name_str.replace(" ",'') + ".jpg")
            plt.close("all")

    #元データから眼電のデータだけを見てみる
# plot_raw_data(df)

##周波数分析をする
def show_fft(eeg_df , ch_list , need_col):
    print("called fft")
    recall_pic = 1 # 周波数スペクトルの欲しい画像
    ch = ch_list
    nd_col = need_col
    # time_step = 0.001 #サンプリング周波数は1 kHZ
    df = eeg_df
    fft_dir = "out/fft_analysis/"
    os.makedirs(fft_dir,exist_ok= True)
    
    
    #データを構成する

    df = df[df["recall_pic_num"] == recall_pic]#特定の画像を想起したときのデータを取り出す

    for temp_recall_total in df["recall_total"].unique():
        temp_recall_total_df = df[df["recall_total"] == temp_recall_total]#1回の想起を取り出す
        for temp_ch_index in range(len(ch)):#特定のチャンネルデータを取り出す
            print("try_ch_" + str(temp_recall_total) +"_" +ch[temp_ch_index], end=" ")
            y_data = [] ##フーリエ変換後のデータ
            ch_only_df = temp_recall_total_df[ch[temp_ch_index]]
            y_data = ch_only_df.to_list()
            
            ##FFT
            # Number of sample points
            N = len(ch_only_df) #2000
            # sample spacing
            dt = 1.0 / 1000.0 #plolymate の
            x_data = np.linspace(0.0, N*dt, N, endpoint=False)
            
            y_fft = fft(y_data)
            x_fft = fftfreq(N, dt)[:N//2]
            
            power = 2.0/N * np.abs(y_fft[0:N//2])  
            freqs = x_fft
          
            
            ##元のグラフを表示する
            x_scale = "log"
            save_dir = fft_dir + "raw/" + x_scale + "/"
            os.makedirs(save_dir,exist_ok= True)
            plt.figure()
            plt.subplot(211)
            plt.plot(y_data,'b-', linewidth=1)
            plt.xlabel('Time')
            plt.ylabel('Ydata')
            plt.grid(True)

            plt.subplot(212)
            plt.plot(freqs,power,linewidth = 1)
            plt.yscale("log")
            plt.xlabel('Frequency'+  str(temp_recall_total)+"_" + str(recall_pic) +"_" +str(ch[temp_ch_index]).replace(" ",''))
            plt.ylabel('Power')
            plt.grid(True)
            if (x_scale == "linear"):
                plt.xscale("linear")

            fig_name = "try_pic_ch_" + str(temp_recall_total)+ "_"+str(recall_pic)  +"_" + str(ch[temp_ch_index]).replace(" ",'') + ".jpg"
            plt.savefig(save_dir + fig_name)
            plt.close("all")

            
            ##フィルタをかける
            #使用する周波数を制限する バンドパスフィルタ
            lower_limit = 1
            higher_limit = 100
            numtaps =101
            
            band_pass_filter = signal.firwin(numtaps ,cutoff=[lower_limit,higher_limit] , window="hamming" ,fs = 1/dt,pass_zero=False)
            y_data = signal.lfilter(band_pass_filter,1,y_data)
           
            y_fft = fft(y_data)     
            x_fft = fftfreq(N, dt)[:N//2]
            
            power = 2.0/N * np.abs(y_fft[0:N//2])  
            freqs = x_fft

            if (x_scale == "linear"):
                plt.xscale("linear")
            
            save_dir = fft_dir + "filtered/" + x_scale + "/"
            #フィルタ後の波形
            os.makedirs(save_dir,exist_ok= True)
            plt.figure()
            plt.subplot(211)
            plt.plot(y_data,'b-', linewidth=1)

            plt.xlabel('Time')
            plt.ylabel('Ydata')
            plt.grid(True)
            

            #フィルタ後の周波数画像
            plt.subplot(212)
            #plt.semilogx(freqs, power,'b.-',lw=1)
            # plt.loglog(freqs, power,'b.-',lw=1)#両対数グラフ
            plt.plot(freqs,power,linewidth = 1)
            plt.yscale("log")
            plt.xlabel('Frequency'+  str(temp_recall_total)+"_" + str(recall_pic) +"_" +str(ch[temp_ch_index]))
            plt.ylabel('Power')
            plt.grid(True)
            
            fig_name = "filterd_try_pic_ch_" + str(temp_recall_total)+ "_"+str(recall_pic)  +"_" + str(ch[temp_ch_index]) + ".jpg"
            plt.savefig(save_dir + fig_name.replace(" ",''))
            plt.close("all")

            # print( "calc_fft_"+str(recall_pic) +"_" + str(temp_recall_total)+"_" + str(ch[temp_ch_index]) )



    

    print("fft_finish ")

show_fft(df,data_ch_list,need_col_list)

##眼電位が混入しているかどうかを調べる 
def check_existance_of_blink (dataframe ):
    #眼電位があるものか判別する
    df = dataframe
    # print("called")
    # eog_col_str = " 30-31"
    eog_col_str = " EOG"

    df = df[[eog_col_str,"recall_total"]]#EOGと想起回数のデータを抽出
    
    #標準偏差の計算
    # recall_dispersion_df = pd.DataFrame()
    # temp_recall_list = []
    # stdev_list = []
    # for temp_recall in df["recall_total"].unique():#想起を一個ずつ取り出す
    #     temp_recall_df = df[df["recall_total"] == temp_recall]
    #     v = np.std(temp_recall_df[eog_col_str])
    #     temp_recall_list.append(temp_recall)
    #     stdev_list.append(v)
    
    # recall_dispersion_df["total_recall"] = temp_recall_list
    # recall_dispersion_df["stdev"] = stdev_list

    # plt.figure
    # # plt.plot(recall_dispersion_df["total_recall"],recall_dispersion_df["stdev"])
    # plt.scatter(recall_dispersion_df["total_recall"],recall_dispersion_df["stdev"])
    # labels = recall_dispersion_df["total_recall"]
    # x = recall_dispersion_df["total_recall"]
    # y = recall_dispersion_df["stdev"]
    # for i, label in enumerate(labels):
    #     plt.text(x[i], y[i],label)  
    # os.makedirs("out/remove_eog/",exist_ok=True)
    # plt.savefig("out/remove_eog/recall_total_stdev.png")
    # plt.close("all")

    #分散の計算
    # recall_dispersion_df = pd.DataFrame()
    # temp_recall_list = []
    # var_list = []
    # for temp_recall in df["recall_total"].unique():#想起を一個ずつ取り出す
    #     temp_recall_df = df[df["recall_total"] == temp_recall]
    #     v = np.var(temp_recall_df[eog_col_str])
    #     temp_recall_list.append(temp_recall)
    #     var_list.append(v)
    
    # recall_dispersion_df["total_recall"] = temp_recall_list
    # recall_dispersion_df["var"] = var_list

    # plt.figure　
    # # plt.plot(recall_dispersion_df["total_recall"],recall_dispersion_df["stdev"])
    # plt.scatter(recall_dispersion_df["total_recall"],recall_dispersion_df["var"])
    # labels = recall_dispersion_df["total_recall"]
    # x = recall_dispersion_df["total_recall"]
    # y = recall_dispersion_df["var"]
    # for i, label in enumerate(labels):
    #     plt.text(x[i], y[i],label)  
    # os.makedirs("out/remove_eog/",exist_ok=True)
    # plt.savefig("out/remove_eog/recall_total_var.png")
    # plt.close("all")

    # print(recall_dispersion_df)

    #取り除いたものをで出力する

    #各想起でEOGにおけるデータの幅を出力する
    try_list =[]
    data_width_list =[]
    out_df = pd.DataFrame()
    for temp_total_recall_int in df["recall_total"].unique():
        temp_df = df[df["recall_total"] ==temp_total_recall_int][eog_col_str]
        data_width = temp_df.max() - temp_df.min()
        try_list.append(temp_total_recall_int)
        data_width_list.append(data_width)

    try_sr = pd.Series(try_list)
    data_width_sr = pd.Series(data_width_list)
    out_df = pd.concat([try_sr,data_width_sr],axis=1 )#横に結合

    plt.figure()
    plt.ylabel("data_width ")
    plt.xlabel("try")
    plt.title("各想起のEOGのデータの幅")
    labels = [try_list,data_width_list]
    plt.scatter(out_df[0],out_df[1])
    for index in range(len(labels[0])):
        
        plt.text(labels[0][index],labels[1][index],str(labels[0][index]))

#     labels[1]
# [963.5899999999999, 963.13, 874.5, 957.56, 825.65, 853.51, 722.37, 949.8499999999999, 509.15999999999997, 909.08, 931.76, 944.35, 860.23, 911.91, ...]
# labels[0]
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, ...]
    os.makedirs("out/data_width/",exist_ok=True)
    plt.savefig("out/data_width/all_data.png")
    print("func_end")
    return 0

check_existance_of_blink(df )





    






print("実行終了")