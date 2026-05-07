import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import ids
import torch
from models import TabIDS
from evaluator import predict, get_eval_metric_fn, EarlyStopping, acc_fn
from sklearn.metrics import confusion_matrix, classification_report

kdd_columns = ['duration of the connection', 'protocol', 'network service', 'connection status flag', 'bytes sent from source to destination',
                'bytes sent from destination to source', '1 if source and destination ip/port are the same, else 0', 'number of wrong tcp fragments', 
                'number of urgent packets', 'number of "hot" indicators', 'number of failed login attempts', '1 if successfully logged in, else 0', 
                'number of compromised conditions', '1 if root shell obtained, else 0', '1 if su command attempted, else 0', 'number of root accesses', 
                'number of file creation operations', 'number of shell prompts invoked', 'number of access control file accesses', 'number of outbound commands',
                '1 if login is to a hot account, else 0', '1 if login is as guest user, else 0', 'number of connections to the same destination host', 
                'number of connections to the same service', 'ratio of connections with syn errors', 'ratio of same-service connections with syn errors', 
                'ratio of connections with rej errors', 'ratio of same-service connections with rej errors', 'ratio of connections to the same service', 
                'ratio of connections to different services', 'ratio of same-service connections to different hosts', 'number of connections to the same host', 
                'number of connections to the same service on that host', 'ratio of connections to the same service on the host-level', 
                'ratio of connections to different services  on the host-level', 'ratio of connections from the same source port', 
                'ratio of of same-service connections from different hosts', 'ratio of connections to host with syn errors', 
                'ratio of same-service connections with syn errors on the host-level', 'ratio of connections with rej errors on the host-level', 
                'ratio of same-service connections with rej errors on the host-level', 'label', 'level']

def normalize_df(df):
    # 标签列
    y = df['label']
    df = df.drop(columns=['label'])
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
    df = pd.concat([df, y.reset_index(drop=True)], axis=1)
    return df

def test_model(cat_cols, num_cols, bin_cols, X_test, Y_test, model_path, class_num):
    model = ids.build_classifier(checkpoint = model_path, num_class = class_num)
    model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols})

    #tokenizer = BertTokenizer.from_pretrained('./ckpt/tokenizer')
    y_pre = predict(model, X_test, Y_test, return_loss = False)
    if y_pre.ndim == 1:
        if y_pre.dtype != int and np.all((y_pre >= 0) & (y_pre <= 1)):
            y_pre = (y_pre > 0.5).astype(int)
    else:
        # 多分类情况，取 argmax
        y_pre = np.argmax(y_pre, axis=1)
 
    conf_mat = confusion_matrix(Y_test, y_pre)
    cr = classification_report(Y_test, y_pre)

    acc = acc_fn(Y_test, y_pre)
    print(conf_mat)
    print(acc)

if __name__ == "__main__":
    # 读取数据集
    kdd_train = pd.read_csv('data/nsl-kdd/KDDTrain+.txt', names = kdd_columns)
    kdd_test = pd.read_csv('data/nsl-kdd/KDDTest+.txt', names = kdd_columns)
    kdd = pd.concat([kdd_train, kdd_test], ignore_index=True)
    
    attack_type = {
        'apache2': 1, 'back': 1, 'land': 1, 'neptune': 1,          # dos
        'mailbomb': 1, 'pod': 1, 'processtable': 1, 'smurf': 1,
        'teardrop': 1, 'udpstorm': 1, 

        'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4,'httptunnel': 4, 
        'ps': 4, 'rootkit': 4, 'sqlattack': 4, 'xterm': 4,      # # u2r

        'ftp_write': 3, 'guess_passwd': 3, 'send_mail': 3,'sendmail': 3, 'snmpgetattack': 3,
        'imap': 3, 'multihop': 3, 'named': 3, 'phf': 3,
        'spy': 3, 'snmpguess': 3, 'warezclient': 3,
        'warezmaster': 3, 'xlock': 3, 'xsnoop': 3,     # r2l
        
        'worm': 2, 'nmap': 2, 'ipsweep': 2,'mscan': 2,'portsweep': 2,'saint': 2,'satan': 2,    # probe

        'normal': 0
    }

    attack_bin_type = {
        'apache2': 1, 'back': 1, 'land': 1, 'neptune': 1,          # dos
        'mailbomb': 1, 'pod': 1, 'processtable': 1, 'smurf': 1,
        'teardrop': 1, 'udpstorm': 1, 

        'buffer_overflow': 1, 'loadmodule': 1, 'perl': 1,'httptunnel': 1, 
        'ps': 1, 'rootkit': 1, 'sqlattack': 1, 'xterm': 1,      # # u2r

        'ftp_write': 1, 'guess_passwd': 1, 'send_mail': 1, 'sendmail': 1, 'snmpgetattack': 1,
        'imap': 1, 'multihop': 1, 'named': 1, 'phf': 1,
        'spy': 1, 'snmpguess': 1, 'warezclient': 1,
        'warezmaster': 1, 'xlock': 1, 'xsnoop': 1,     # r2l
        
        'worm': 1, 'nmap': 1, 'ipsweep': 1,'mscan': 1,'portsweep': 1,'saint': 1,'satan': 1,    # probe

        'normal': 0
    }
    # attack_bin_type = { 1:1, 2:1, 3:1, 4:1, 0:0}  # 映射为二分类标签
    kdd.columns = kdd_columns
    # kdd_test.columns = kdd_columns
    
    # 训练集处理
        # 映射标签
    kdd['label'] = kdd['label'].map(attack_bin_type)
    
    # print(kdd.iloc[0:5])  # 打印前5行数据
    
    dos = kdd[kdd['label'] == 1]  # dos攻击
    dos = dos.sample(n = 5000, random_state = 42)
    
    # probe = kdd[kdd['label'] == 2]  # probe
    # probe = probe.sample(n = 5000, random_state = 42)

    normal = kdd[kdd['label'] == 0]  # 正常流量
    normal = normal.sample(n = 5000, random_state = 42)
    
    # # 将数据集分为易攻击和难攻击
    # kdd_easy = kdd[kdd['level']<=10]
    # kdd_hard = kdd[kdd['level']>10]、
    kdd = kdd[kdd['level'] >= 10 ]
    # kdd_test = kdd_test[kdd_test['level'] >= 10 ]
    
    # 训练集使用的数据
    # kdd = pd.concat([dos, kdd[kdd['label'] == 3], normal, probe], ignore_index=True)  #kdd[kdd['label'] == 2], 
    print(kdd['label'].value_counts())


    # 测试集处理
    # unmapped = kdd_test[~kdd_test['label'].isin(attack_type.keys())]['label'].unique()
    # print("未映射标签：", unmapped)
    # kdd_test['label'] = kdd_test['label'].map(attack_type)  

    # print(kdd_test['label'].value_counts())
    
    # dos = kdd_test[kdd_test['label'] == 1]  # dos攻击
    # dos = dos.sample(n = 500, random_state = 42)
    
    # probe = kdd_test[kdd_test['label'] == 2]  # probe
    # probe = probe.sample(n = 500, random_state = 42)
    
    # r2l = kdd_test[kdd_test['label'] == 3]  # r2l
    # r2l = r2l.sample(n = 500, random_state = 42)

    # normal = kdd_test[kdd_test['label'] == 0]  # 正常流量
    # normal = normal.sample(n = 500, random_state = 42)
    
    # # 测试集使用的数据
    # kdd_test = pd.concat([dos, r2l, normal, probe], ignore_index=True)   # kdd_test[kdd_test['label'] == 2],
    
    # print(kdd_easy.count())
    # print(kdd_hard.count())

    # 全数值的嵌入
    # protocol_encoder = LabelEncoder()
    # service_encoder = LabelEncoder()
    # flag_encoder = LabelEncoder()
    # kdd['protocol'] = protocol_encoder.fit_transform(kdd['protocol'])
    # kdd['network service'] = service_encoder.fit_transform(kdd['network service'])
    # kdd['connection status flag'] = flag_encoder.fit_transform(kdd['connection status flag'])
    # kdd_test['protocol_type'] = protocol_encoder.fit_transform(kdd_test['protocol_type'])
    # kdd_test['service'] = service_encoder.fit_transform(kdd_test['service'])
    # kdd_test['flag'] = flag_encoder.fit_transform(kdd_test['flag'])
    
    flag_mapping = {
    'S0': 'connection attempt seen, no reply',
    'S1': 'connection established, not terminated',
    'S2': 'connection established and close attempt by originator seen',
    'S3': 'connection established and close attempt by responder seen',
    'SF': 'normal data and termination',
    'REJ': 'connection attempt rejected',
    'RSTO': 'connection reset by the originator',
    'RSTR': 'connection reset by the responder',
    'RSTOS0': 'originator sent a syn followed by a rst, wenever saw a syn-ack fromthe responder',
    'SH': 'originator sent a syn followed by a fin, we never saw a syn-ack from the responder',
    'OTH': 'no SYN seen, just midstream traffic',
    'SHR': 'responder sent a syn-ack followed by a fin, wenever saw a syn from the originator'
    }

    # 替换DataFrame中的符号为描述
    kdd['connection status flag'] = kdd['connection status flag'].map(flag_mapping)

    
    # 映射二进制标签
    # kdd['label'] = kdd['label'].map(attack_bin_type)
    kdd.drop(columns = 'level', inplace=True)  # 删除level列
    # kdd_test.drop(columns = 'level', inplace=True)  # 删除level列
    
    
    # 归一化处理
    # 提取标签
    y_train = kdd['label']
    # y_test = kdd_test['label']

    # 去掉 label 列，获取特征
    X_train = kdd.drop(columns=['label'])
    # X_test = kdd_test.drop(columns=['label'])

    # 选择数值型列
    num_cols = X_train.select_dtypes(include='number').columns.tolist()

    # 用训练集拟合 MinMaxScaler
    scaler = MinMaxScaler().fit(X_train[num_cols])
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

    # 使用相同的 scaler 归一化测试集
    # X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 拼回标签
    kdd_norm = X_train.copy()
    kdd_norm['label'] = y_train.values

    # kdd_test_norm = X_test.copy()
    # kdd_test_norm['label'] = y_test.values
    # kdd_norm = normalize_df(kdd)  # 只对正常流量和攻击流量进行归一化处理
    # kdd_test_norm = normalize_df(kdd_test)
    
    
    # 预处理

    

    # X_test = kdd_test_norm.iloc[:, :-1]
    # Y_test = kdd_test_norm["label"]
    # X_train, X_val, Y_train, Y_val = train_test_split(kdd_norm.iloc[:, :-1], kdd_norm["label"], test_size=0.20, shuffle=True, stratify=kdd_norm["label"])
    
    
    # kdd['label'] = kdd['label'].map(attack_bin_type)  # 映射二进制标签
    # print(kdd.iloc[0:5])
    # print(np.isnan(kdd).any().sum())
    # print(np.isinf(kdd).any().sum())
    
    # 划分训练集、验证集和测试集
    xtrain, X_test, ytrain, Y_test = train_test_split(kdd.iloc[:, :-1], kdd["label"],test_size=0.20,shuffle=True,stratify = kdd["label"])
    X_train, X_val, Y_train, Y_val = train_test_split(xtrain, ytrain,test_size=0.1,shuffle=True, stratify = ytrain)
    label_encoder = LabelEncoder()
    encode_Y_train = pd.Series(label_encoder.fit_transform(Y_train), index=Y_train.index)
    encode_Y_test = pd.Series(label_encoder.fit_transform(Y_test), index=Y_test.index)
    encode_Y_val = pd.Series(label_encoder.fit_transform(Y_val), index=Y_val.index)
    train_classes = len(Y_train)
    test_classes = len(Y_test)
    val_classes = len(Y_val)
    print(f"Number of train labels: {train_classes}")
    print(f"Number of test labels: {test_classes}")
    print(f"Number of val labels: {val_classes}")
    
    print(Y_train.value_counts())
    
    if os.path.exists('/home/zhangheming/code/code/feature_type/kdd/numerical_feature_kdd.txt'):
        with open('/home/zhangheming/code/code/feature_type/kdd/numerical_feature_kdd.txt', 'r') as f:
            num_cols = [x.strip().lower() for x in f.readlines()]
    else:
        num_cols = []
    if os.path.exists('binary_feature.txt'):
        with open('binary_feature.txt', 'r') as f:
            bin_cols = [x.strip().lower() for x in f.readlines()]
    else:
        bin_cols = []
    if os.path.exists('/home/zhangheming/code/code/feature_type/kdd/cat_feature_kdd.txt'):
        with open('/home/zhangheming/code/code/feature_type/kdd/cat_feature_kdd.txt', 'r') as f:
            cat_cols = [x.strip().lower() for x in f.readlines()]
    else:
        cat_cols = []
    # cat_cols = [col for col in X_train.columns if col not in num_cols and col not in bin_cols]
    
    # kdd['label'] = kdd['label'].apply(lambda x: x.split('.')[0])  # 去掉最后的点和数字
    print("label", kdd['label'].value_counts())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build classifier
    #model=ids.train()
    print("X_train", X_train[0:3])
    print("X_test", X_test[0:3])
 
    # 多分类测试
    # model = ids.build_classifier(cat_cols, num_cols, bin_cols, num_class=4)    ###
    # model.to(device)
    
    
    # ids.train(model, (X_train,Y_train), (X_val,Y_val), lr = 0.0001, output_dir='./ckpt_kdd', num_epoch = 10)
    
    # # attn = model.last
    
    # test_model(cat_cols, num_cols, bin_cols, X_test, Y_test, './ckpt_kdd', class_num = 4)
    
    
    # 二分类测试
    model = ids.build_classifier(cat_cols, num_cols, bin_cols, num_class=2)    ###
    model.to(device)
    
    
    ids.train(model, (X_train,Y_train), (X_val,Y_val), lr = 0.00001, output_dir='./ckpt_bin_kdd', num_epoch = 50)
    
    # attn = model.last
    
    test_model(cat_cols, num_cols, bin_cols, X_test, Y_test, './ckpt_bin_kdd', class_num = 2)