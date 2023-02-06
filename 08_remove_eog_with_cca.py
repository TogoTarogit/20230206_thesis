##
import pandas as pd 
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "MS Gothic"
from scipy import fftpack
from sklearn.decomposition import FastICA

##データの読みだし
test_csv_path = r""
df_path = test_csv_path
df = pd.read_csv(df_path)
#print(df.columns)
# ['TIME', ' Fp1-REF', ' Fp2-REF', ' F3-REF', ' F4-REF', ' C3-REF', ' C4-REF', ' P3-REF', ' P4-REF', ' O1-REF', ' O2-REF', ' F7-REF',
# ' F8-REF', ' T3-REF', ' T4-REF', ' T5-REF', ' T6-REF', ' Fz-REF',
# ' Cz-REF', ' Pz-REF', ' 30-31', ' A1-REF', ' A2-REF', ' EXT',
# 'recall_pic_num', 'recall_total']

eeg_ch_col = [' Fp1-REF', ' Fp2-REF', ' F3-REF', ' F4-REF', ' C3-REF', ' C4-REF', ' P3-REF', ' P4-REF', ' O1-REF', ' O2-REF', ' F7-REF',
' F8-REF', ' T3-REF', ' T4-REF', ' T5-REF', ' T6-REF', ' Fz-REF',
' Cz-REF', ' Pz-REF', ' A1-REF', ' A2-REF', ]
eog_ch_col = [' 30-31']
other_col = ['TIME',  ' EXT', 'recall_pic_num', 'recall_total']

##正準相関分析を用いて眼電除去を試みる
def remove_eog_with_cca(dataframe,eeg_ch,other_col):
    df = dataframe
    eeg_ch_col = eeg_ch
    eog_ch_col = [' 30-31']
    df = df[df["recall_total"]==1]
    from sklearn.cross_decomposition import CCA
    X = df[eog_ch_col].to_numpy().flatten().reshape(-1,1)

    Y = df[eeg_ch_col[0]].to_numpy()
    cca = CCA(n_components=1)
    cca.fit(X, Y)
    X_c,Y_c = cca.transform(X,Y)

    print(X_c)
    print(Y_c)
    os.makedirs("out/cca/",exist_ok=True)
    plt.figure()
    plt.plot(X_c)
    plt.savefig("out/cca/" + "X_c" + ".png")
    plt.figure()
    plt.plot(Y_c)
    plt.savefig("out/cca/" + "Y_c" + ".png")


# remove_eog_with_cca(df,eeg_ch_col,other_col)

def remove_eog_with_ica(dataframe_1,eeg_ch_col_1,other_ch):
    # https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html#sphx-glr-auto-examples-decomposition-plot-ica-blind-source-separation-py
    print("called " + remove_eog_with_ica.__name__)
    df = dataframe_1
    eeg_ch_list = eeg_ch_col_1
    other_ch_list = other_ch
    
    save_dir = "out/apply_ica/"#結果保存場所を作成
    os.makedirs( save_dir,exist_ok=True)

    for temp_recall_int in df["recall_total"].unique():
        temp_df = df[df["recall_total"]==temp_recall_int]##特定の想起データを取り出す
        recall_pic_int = temp_df["recall_pic_num"].iloc[0]#想起画像を抽出
        temp_df = temp_df[eeg_ch_list]##df の中からeegだけのデータにする
        temp_df_np = temp_df.to_numpy()
        ica = FastICA(max_iter=2001,whiten='unit-variance')
        df_new_np = ica.fit_transform(temp_df_np)
        
        #描画処理
        cols = []#列の名前を作成する
        for index in range(len(df_new_np[0])):
            cols.append("ica" + str(index+1))

        view_df = pd.DataFrame(df_new_np , columns=cols)
        
    
        plt.figure()
        plt.plot(view_df,lw = 0.5)
        plt.title("total_pic_" + str(temp_recall_int) +"_"+ str(recall_pic_int))
        plt.savefig(save_dir + "total_pic_" + str(temp_recall_int)+ "_" + str(recall_pic_int) +".png" )
        plt.close('all')       
        





    
remove_eog_with_ica(df,eeg_ch_col,other_col)

print("finish")