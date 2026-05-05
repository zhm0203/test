import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import ids
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from evaluator import predict, get_eval_metric_fn, EarlyStopping, acc_fn
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertForSequenceClassification
from models import TabIDS, TabIDSFeatureExtractor, TabIDSFeatureProcessor
from models import TabIDSInputEncoder, TabIDSModel


cic2018_columns=['destination port','protocol','flow duration','total forward packets',
                 'total backward packets','total length of forward packets','total length of backward packets',
                 'maximum length of forward packet',
                 'minimum length of forward packet','average length of forward packets',
                 'standard deviation of forward packet length','maximum length of backward packet',
                 'minimum length of backward packet','average length of backward packets',
                 'standard deviation of backward packet length','flow bytes per second','flow packets per second',
                 'average value of flow inter-arrival time','standard deviation of flow inter-arrival time',
                 'maximum value of flow inter-arrival time','minimum value of flow inter-arrival time',
                 'total value of forward flow inter-arrival time','average value of forward flow inter-arrival time',
                 'standard deviation of forward flow inter-arrival time','maximum value of forward flow inter-arrival time',
                 'minimum value of forward flow inter-arrival time','total value of backward flow inter-arrival time',
                 'average value of backward flow inter-arrival time','standard deviation of backward flow inter-arrival time',
                 'maximum value of backward flow inter-arrival time','minimum value of backward flow inter-arrival time',
                 'forward psh flags','backward psh flags','forward urg flags','backward urg flags','forward header length',
                 'backward header length','forward packets per second','backward packets per second','minimum length of packet',
                 'maximum length of packet','average length of packets','standard deviation of packets length',
                 'variance of packet length','number of fin flag','number of syn flag','number of rst flag','number of psh flag',
                 'number of ack flag','number of urg flag','number of cwe flag','number of ece flag',
                 'ratio of backward flow to forward flow','average packet size','average forward segment size',
                 'average backward segment size','forward average bytes per bulk',
                 'forward average packets per bulk', 'forward average bulk rate','backward average bytes per bulk',
                 'backward average packets per bulk','backward average bulk rate','subflow forward packets','subflow forward bytes',
                 'subflow backward packets','subflow backward bytes','forward initial window bytes count',
                 'backward initial window bytes count','number of active forward packets','minimum size of forward segment',
                 'average value of active','standard deviation of active', 'maximum value of active','minimum value of active',
                 'average value of idle','idle standard deviation','maximum value of idle','minimum value of idle','label'
                 ]


def normalize_df(df):
    # 获取所有数值列
    num_cols = df.select_dtypes(include='number').columns.tolist()
    
    # 归一化计算
    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # 如果最大值和最小值相等，跳过该列
        if col_max == col_min:
            # print(f"跳过列 {col}，因为所有值相同")
            continue
        
        # 归一化
        df[col] = (df[col] - col_min) / (col_max - col_min)
        
    return df


def test_model(cat_cols, num_cols, bin_cols, X_test, Y_test, model_path):
    # model = TabIDS(
    #     categorical_columns=cat_cols,
    #     numerical_columns=num_cols,
    #     binary_columns=bin_cols,
    #     num_class=7
    # )
    # print(model_path)
    # model.load(model_path)
    model = ids.build_classifier(checkpoint = model_path)
    model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols})
    y_pre = predict(model, X_test, Y_test, return_loss = False)
    # conf_mat = confusion_matrix(Y_test, y_pred)
    # cr = classification_report(Y_test, y_pred)
    # # print(conf_mat)
    # print(cr)
    # conf_mat = confusion_matrix(Y_test, y_pred)
    # print(conf_mat)
    if y_pre.ndim == 1:
        if y_pre.dtype != int and np.all((y_pre >= 0) & (y_pre <= 1)):
            y_pre = (y_pre > 0.5).astype(int)
    else:
        # 多分类情况，取 argmax
        y_pre = np.argmax(y_pre, axis=1)
    # y_pre = np.argmax(y_pre, -1)
    conf_mat = confusion_matrix(Y_test, y_pre)
    cr = classification_report(Y_test, y_pre, digits=4)
    print(cr)

    acc = acc_fn(Y_test, y_pre)
    # print(conf_mat)
    print(acc)


def fine_tuneing(cat_cols, num_cols, bin_cols, X_train, X_val, Y_train, Y_val, X_test, Y_test, model_path):
    # model = TabIDS(
    #     categorical_columns=cat_cols,
    #     numerical_columns=num_cols,
    #     binary_columns=bin_cols,
    #     num_class=7
    # )
    model = ids.build_classifier(checkpoint = model_path, num_class = 7)
    print(model_path)
    model.load(model_path)
    model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols})  # 修改 num_class 为 5

    # 模型微调
    # num_classes = len(Y_train.unique())   # 改变原始模型的输出类别
    # print('Y_train:',len(Y_train.unique()))
    # model.fc = nn.Linear(128, num_classes)     # 128是隐藏层输出维度，
    ids.train(model, (X_train,Y_train), (X_val,Y_val), lr=1e-4, num_epoch=20, output_dir='./ckpt_fine_tuneing_8.6')
    
    # y_prob = np.array(p).reshape(-1)
    # y_pred = (y_prob >= 0.5).astype(int)  # 二分类阈值为 0.5

# 这里下午改
    y_pre = predict(model, X_test, Y_test, return_loss = False)
    y_pred = np.argmax(y_pre, -1)
    # conf_mat = confusion_matrix(Y_test, y_pred)
    cr = classification_report(Y_test, y_pred, digits=4)
    # print(conf_mat)
    print(cr)

    conf_mat = confusion_matrix(Y_test, y_pred)
    classes = np.unique(Y_test)
    cm_df = pd.DataFrame(conf_mat, index=classes, columns=classes)
    print(cm_df)





if __name__ == '__main__':
    
    # # 整合2018数据
    # data_1 = pd.read_csv("./data/archive/02-14-2018.csv", index_col=0,low_memory=False)   
    # print(data_1['Label'].value_counts())
    # benign = data_1[data_1['Label'] == 'Benign']
    # benign = benign.sample(n = 5000, random_state = 42)
    # ftp = data_1[data_1['Label'] == 'FTP-BruteForce']
    # ftp = ftp.sample(n = 5000, random_state = 42)
    # ssh = data_1[data_1['Label'] == 'SSH-Bruteforce']
    # ssh = ssh.sample(n = 5000, random_state = 42)
    
    # data_2 = pd.read_csv("./data/archive/02-15-2018.csv", index_col=0,low_memory=False)
    # print(data_2['Label'].value_counts())
    # goldeneye = data_2[data_2['Label'] == 'DoS attacks-GoldenEye']
    # goldeneye = goldeneye.sample(n = 5000, random_state = 42)
    # slowloris = data_2[data_2['Label'] == 'DoS attacks-Slowloris']
    # slowloris = slowloris.sample(n = 5000, random_state = 42)
    
    # data_3 = pd.read_csv("./data/archive/02-16-2018.csv", index_col=0,low_memory=False)
    # print(data_3['Label'].value_counts())
    # hulk = data_3[data_3['Label'] == 'DoS attacks-Hulk']
    # hulk = hulk.sample(n = 5000, random_state = 42)
    # SlowHTTPTest = data_3[data_3['Label'] == 'DoS attacks-SlowHTTPTest']
    # SlowHTTPTest = SlowHTTPTest.sample(n = 5000, random_state = 42)

    # # data_4 = pd.read_csv("./data/archive/02-20-2018.csv", index_col=0,low_memory=False)  ## 83
    # # # print(data_4['Label'].value_counts())
    # # # print(len(data_4.columns),data_4.columns)
    # # loichttp = data_4[data_4['Label'] == 'DDoS attacks-LOIC-HTTP']
    # # loichttp = loichttp.sample(n = 5000, random_state = 42)
    # # loichttp.drop('Src IP', axis=1, inplace=True)
    # # loichttp.drop('Src Port', axis=1, inplace=True)
    # # loichttp.drop('Dst IP', axis=1, inplace=True)
    # # loichttp.drop('Dst Port', axis=1, inplace=True)
    # # loichttp = loichttp.reset_index(drop=True)
    # # # loichttp[:2].to_csv('loic.csv')
    # # # print(loichttp.columns)
    # # # sys.exit()
    

    

    # # data_5 = pd.read_csv("./data/archive/02-21-2018.csv", index_col=0,low_memory=False)
    # # print(data_5['Label'].value_counts())
    # # hoic = data_5[data_5['Label'] == 'DDOS attack-HOIC']
    # # hoic = hoic.sample(n = 5000, random_state = 42)
    # # loicudp = data_5[data_5['Label'] == 'DDOS attack-LOIC-UDP']
    # # loicudp = loicudp.sample(n = 1730, random_state = 42)


    # # #data_6 = pd.read_csv("./data/archive/02-22-2018.csv", index_col=0,low_memory=False)

    # # #data_7 = pd.read_csv("./data/archive/02-23-2018.csv", index_col=0,low_memory=False)

    # data_8 = pd.read_csv("./data/archive/02-28-2018.csv", index_col=0,low_memory=False)

    # Infilteration = data_8[data_8['Label'] == 'Infilteration']
    # Infilteration = Infilteration.sample(n = 5000, random_state = 42)   
    
    # # #data_9 = pd.read_csv("./data/archive/03-01-2018.csv", index_col=0,low_memory=False)

    # data_10 = pd.read_csv("./data/archive/03-02-2018.csv", index_col=0,low_memory=False)

    # print(data_10['Label'].value_counts())
    # Bot = data_10[data_10['Label'] == 'Bot']
    # Bot = Bot.sample(n = 5000, random_state = 42)   

    # # data2018 = pd.concat([benign 0, ftp 1, ssh 2, goldeneye 3, slowloris 4, hulk 5, SlowHTTPTest 6, loichttp 7, hoic, loicudp, Infilteration, Bot])
    
    
    # # data2018 = pd.concat([benign, ftp, ssh, goldeneye, slowloris, hulk, SlowHTTPTest, Infilteration, Bot])
    # data2018 = pd.concat([benign, ftp, ssh, goldeneye, hulk, SlowHTTPTest, Bot])
    # print(data2018['Label'].value_counts())
    # data2018.to_csv("cic2018.csv")
    # sys.exit()



    

    cic2018 = pd.read_csv("./cic2018.csv")
    # cic2018.drop('Src IP', axis=1, inplace=True)
    # cic2018.drop('Src Port', axis=1, inplace=True)
    # cic2018.drop('Dst IP', axis=1, inplace=True)
    # cic2018.drop('Dst Port', axis=1, inplace=True)
    cic2018.drop('Timestamp', axis=1, inplace=True)
    print(cic2018['Label'].value_counts())
    print(len(cic2018_columns))
    


    cic2018.columns=cic2018_columns

        # # 去掉全0的列
    # cic2018.loc[:, (cic2018 != 0).any(axis=0)]
    

    # 去掉inf，将inf替换为NaN
    cic2018.replace(np.inf, np.nan, inplace=True) 

    # 处理NaN值，将含有NaN的行都删掉
    cic2018.replace(['', None], np.nan, inplace=True)
    cic2018 = cic2018.fillna(method='ffill')

    cic2018.drop(columns=cic2018.columns[(cic2018 == 0).all()])

    # 归一化
    norm2018 = normalize_df(cic2018)
    cic2018.to_csv('check.csv')

    

    # 数据处理之后的统计值 
    #status=norm2018.describe()
    #status.to_csv('status_norm.csv')
    # benign = norm2018[norm2018['label'] == 'Benign']
    # benign = benign.sample(n = 10000, random_state = 42)

    # ftp = norm2018[norm2018['label'] == 'FTP-BruteForce']
    # ftp = ftp.sample(n = 10000, random_state = 42)

    # ssh = norm2018[norm2018['label'] == 'SSH-Bruteforce']
    # ssh = ssh.sample(n = 10000, random_state = 42)

    # cic2018_choose = pd.concat([benign, ftp, ssh])


    # print(cic2018_choose['label'].value_counts())



    #attack_ = data_total[data_total['Label'] != 'BENIGN']

    #print(attack_['Label'].value_counts())

    # 划分训练集、验证集、测试集
    xtrain, X_test, ytrain, Y_test = train_test_split(norm2018.iloc[:, :-1], norm2018["label"],test_size=0.6,shuffle=True)
    X_train, X_val, Y_train, Y_val = train_test_split(xtrain, ytrain,test_size=0.1,shuffle=True)

    label_encoder = LabelEncoder()
    encode_Y_train = pd.Series(label_encoder.fit_transform(Y_train), index=Y_train.index)
    encode_Y_test = pd.Series(label_encoder.fit_transform(Y_test), index=Y_test.index)
    encode_Y_val = pd.Series(label_encoder.fit_transform(Y_val), index=Y_val.index)
    train_classes = len(set(encode_Y_train))
    test_classes = len(set(encode_Y_test))
    val_classes = len(set(encode_Y_val))
    print(f"Number of train labels: {train_classes}")
    print(f"Number of test labels: {test_classes}")
    print(f"Number of val labels: {val_classes}")


    # 二分类标签映射
    label_map = {'FTP-BruteForce': 1, 'SSH-Bruteforce': 1, 'Bot': 1, 'Benign': 0, 'DoS attacks-GoldenEye': 1, 'DoS attacks-SlowHTTPTest': 1, 'DoS attacks-Hulk': 1}
    bin_y_train = Y_train.map(label_map)
    bin_y_val = Y_val.map(label_map)
    bin_y_test = Y_test.map(label_map)

    print("原始标签 classes_：", Y_train.unique())
    print("标签 → 编码 映射：")
    for i, label in enumerate(Y_train.unique()):
        print(f"{label} → {i}")

    # 多分类标签映射
    mal_label_map = {'FTP-BruteForce': 0, 'SSH-Bruteforce': 1, 'Bot': 2, 'Benign': 3, 'DoS attacks-GoldenEye': 4, 'DoS attacks-SlowHTTPTest': 5, 'DoS attacks-Hulk': 6}
    mal_ytrain = Y_train.map(mal_label_map)
    mal_yval = Y_val.map(mal_label_map)
    mal_ytest = Y_test.map(mal_label_map)


    if val_classes != train_classes or test_classes!= train_classes:
        sys.exit()

    print("train:",len(X_train)/len(norm2018.index)*100,"%")
    print("test:",len(X_test)/len(norm2018.index)*100,"%")
    print("val:",len(X_val)/len(norm2018.index)*100,"%")
    
    # load dataset by specifying dataset name
    #allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_NSLKDD()
    
    if os.path.exists('numerical_feature_2018.txt'):
        with open('numerical_feature_2018.txt', 'r') as f:
            num_cols = [x.strip().lower() for x in f.readlines()]
    else:
        num_cols = []
    if os.path.exists('binary_feature.txt'):
        with open('binary_feature.txt', 'r') as f:
            bin_cols = [x.strip().lower() for x in f.readlines()]
    else:
        bin_cols = []
    cat_cols = [col for col in X_train.columns if col not in num_cols and col not in bin_cols]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build classifier
    # #model=ids.train()
    model = ids.build_classifier(cat_cols, num_cols, bin_cols, num_class = 7)    ###
    model.to(device)

    # 微调多分类
    fine_tuneing(cat_cols, num_cols, bin_cols, X_train, X_val, mal_ytrain, mal_yval, X_test, mal_ytest, './multi_more_2017')
    
    # 微调二分类
    # fine_tuneing(cat_cols, num_cols, bin_cols, X_train, X_val, bin_y_train, bin_y_val, X_test, bin_y_test, './bin_2017')
    
    # # 监督学习
    # ids.train(model, (X_train,bin_y_train), (X_val,bin_y_val), lr = 0.001, num_epoch=10, output_dir='./ckpt_bin_2018')
    # test_model(cat_cols, num_cols, bin_cols, X_test, bin_y_test, './ckpt_bin_2018')
    # sys.exit()
    # print("标签分布:", encode_Y_test.value_counts())

 
    # model = ids.build_classifier(cat_cols, num_cols, bin_cols, num_class = 7)    ###
    # model.to(device)
    # print(model.device)
    # # print("num_cols:",len(num_cols))
    # # print('X_train:', X_train.shape)

    # # start training
    # ids.train(model, (X_train,bin_y_train), (X_val,bin_y_val))

    # test_model(cat_cols, num_cols, bin_cols, X_test, encode_Y_test, './ckpt')
