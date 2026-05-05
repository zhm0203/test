import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dataset import load_2017
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import shap


cic2017_columns=['destination port','flow duration','total forward packets','total backward packets',
'total length of forward packets','total length of backward packets','maximum length of forward packet','minimum length of forward packet',
'average length of forward packets','standard deviation of forward packet length','maximum length of backward packet','minimum length of backward packet',
'average length of backward packets','standard deviation of backward packet length','flow bytes per second','flow packets per second',
'average value of flow inter-arrival time','standard deviation of flow inter-arrival time','maximum value of flow inter-arrival time',
'minimum value of flow inter-arrival time','total value of forward flow inter-arrival time','average value of forward flow inter-arrival time',
'standard deviation of forward flow inter-arrival time','maximum value of forward flow inter-arrival time','minimum value of forward flow inter-arrival time',
'total value of backward flow inter-arrival time','average value of backward flow inter-arrival time','standard deviation of backward flow inter-arrival time',
'maximum value of backward flow inter-arrival time','minimum value of backward flow inter-arrival time','forward psh flags','backward psh flags',
'forward urg flags','backward urg flags','forward header length','backward header length','forward packets per second',
'backward packets per second','minimum length of packet','maximum length of packet','average length of packets','standard deviation of packets length',
'variance of packet length','number of fin flag','number of syn flag','number of rst flag','number of psh flag','number of ack flag',
'number of urg flag','number of cwe flag','number of ece flag','ratio of backward flow to forward flow','average packet size',
'average forward segment size','average backward segment size','length of forward header','forward average bytes per bulk',
'forward average packets per bulk','forward average bulk rate','backward average bytes per bulk','backward average packets per bulk',
'backward average bulk rate','subflow forward packets','subflow forward bytes','subflow backward packets','subflow backward bytes',
'forward initial window bytes count','backward initial window bytes count','number of active forward packets','minimum size of forward segment',
'average value of active','standard deviation of active','maximum value of active','minimum value of active','average value of idle',
'idle standard deviation','maximum value of idle','minimum value of idle','label']

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
    model = TabIDS(
        categorical_columns=cat_cols,
        numerical_columns=num_cols,
        binary_columns=bin_cols,
        num_class=class_num,
    )
    print(model_path)
    model.load(model_path)
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




if __name__ == '__main__':
    
    """data_monday = pd.read_csv("D:\TabIDS\code\code\data\MachineLearningCSV\MachineLearningCVE\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", index_col=0)   
    data_friday1 = pd.read_csv("D:\TabIDS\code\code\data\MachineLearningCSV\MachineLearningCVE\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", index_col=0)
    data_friday2 = pd.read_csv("D:\TabIDS\code\code\data\MachineLearningCSV\MachineLearningCVE\Friday-WorkingHours-Morning.pcap_ISCX.csv", index_col=0)
    data_friday3 = pd.read_csv("D:\TabIDS\code\code\data\MachineLearningCSV\MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv", index_col=0)
    data_thursday1 = pd.read_csv("D:\TabIDS\code\code\data\MachineLearningCSV\MachineLearningCVE\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", index_col=0)
    data_thursday2 = pd.read_csv("D:\TabIDS\code\code\data\MachineLearningCSV\MachineLearningCVE\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", index_col=0)
    data_tuesday = pd.read_csv("D:\TabIDS\code\code\data\MachineLearningCSV\MachineLearningCVE\Tuesday-WorkingHours.pcap_ISCX.csv", index_col=0)
    data_wednesday = pd.read_csv("D:\TabIDS\code\code\data\MachineLearningCSV\MachineLearningCVE\Wednesday-workingHours.pcap_ISCX.csv", index_col=0)
    # 数据集的所有数据
    data_total = pd.concat([data_monday,data_friday1,data_friday2,data_friday3,data_thursday1,data_thursday2,data_tuesday,data_wednesday])

    data_total.to_csv('cic2017_tatal.csv')"""


    # cic2017=pd.read_csv("./cic2017_total.csv")

    # cic2017.columns=cic2017_columns
    # # 得到标签列索引
    # #print(cic2017['Label'].value_counts())

    # #status=cic2017.describe()
    
    # #print(status)
    # # 去掉全0的列
    # cic2017.loc[:, (cic2017 != 0).any(axis=0)]

    # # 去掉inf，将inf替换为NaN
    # #cic2017.replace(np.inf, np.nan, inplace=True) 

    # # 处理NaN值，将含有NaN的行都删掉
    # # cic2017.dropna(inplace=True)
    # cic2017.fillna(0, inplace=True)  # 或者用0填充NaN

    # # 归一化
    # cic2017=normalize_df(cic2017)

    # # 数据处理之后的统计值 
    # #status=cic2017.describe()
    # #status.to_csv('status_norm.csv')


    # print(cic2017['label'].value_counts())
    # # 筛选出标签为 target_label 的数据
    # dos_hulk = cic2017[cic2017['label'] == 'DoS Hulk']
    # dos_hulk = dos_hulk.sample(n=5000, random_state=42)

    # Bot = cic2017[cic2017['label'] == 'Bot']
    # Bot = Bot.sample(n=1966, random_state=42)

    # Web = cic2017[cic2017['label'] == 'Web Attack � Brute Force']
    # Web = Web.sample(n=1507, random_state=42)

    # DoS_GoldenEye = cic2017[cic2017['label'] == 'DoS GoldenEye']
    # DoS_GoldenEye = DoS_GoldenEye.sample(n=5000, random_state=42)

    # FTP_Patator = cic2017[cic2017['label'] == 'FTP-Patator']
    # FTP_Patator = FTP_Patator.sample(n=5000, random_state=42)

    # SSH_Patator = cic2017[cic2017['label'] == 'SSH-Patator']
    # SSH_Patator = SSH_Patator.sample(n=5000, random_state=42)

    # DoS_slowloris = cic2017[cic2017['label'] == 'DoS slowloris']
    # DoS_slowloris = DoS_slowloris.sample(n=5000, random_state=42)

    # DoS_Slowhttptest = cic2017[cic2017['label'] == 'DoS Slowhttptest']
    # DoS_Slowhttptest = DoS_Slowhttptest.sample(n=5000, random_state=42)

    # BENIGN = cic2017[cic2017['label'] == 'BENIGN']
    # BENIGN = BENIGN.sample(n=5000, random_state=42)

    # # cic_choose = pd.concat([dos_hulk, Bot, Web, DoS_GoldenEye, FTP_Patator, SSH_Patator, DoS_slowloris, DoS_Slowhttptest, BENIGN])   # 9类
    # cic_choose = pd.concat([Bot, Web, DoS_GoldenEye, FTP_Patator, DoS_slowloris, DoS_Slowhttptest, BENIGN])  # 7类

    # # cic_choose = pd.concat([BENIGN, FTP_Patator, SSH_Patator])   # 训三类

    # print(cic_choose['label'].value_counts())

    # cic_choose.to_csv('cic_choose.csv')

##################################################

    cic2017 = pd.read_csv("./cic_choose.csv", index_col=0)  # 读取数据集
    print(cic2017['label'].value_counts())
    print(cic2017[0:5])
    # sys.exit()


    label_map = {'DoS Slowhttptest': 1, 'BENIGN': 0, 'DoS GoldenEye': 1, 'Bot': 1, 'DoS Hulk': 1, 'Web Attack � Brute Force': 1, 'FTP-Patator': 1, 'DoS slowloris': 1, 'SSH-Patator': 1}
    cic2017['label'] = cic2017['label'].map(label_map)
    print(cic2017['label'].value_counts())
    # print(bin_y_train.value_counts())
    mal = cic2017[cic2017['label'] == 1]
    mal = mal.sample(n=5000, random_state=42)
    cic2017 = pd.concat([mal, cic2017[cic2017['label'] == 0]], ignore_index=True)  # 只保留二分类的恶意流量和正常流量
    print("NaN in bin_y_val:", np.isnan(cic2017['label']).sum())
    # sys.exit()


    # 划分训练集、验证集、测试集
    xtrain, X_test, ytrain, Y_test = train_test_split(cic2017.iloc[:, :-1], cic2017["label"],test_size=0.20,shuffle=True,stratify = cic2017["label"])
    X_train, X_val, Y_train, Y_val = train_test_split(xtrain, ytrain,test_size=0.1,shuffle=True, stratify = ytrain)
    #print(Y_val)

    # label_encoder = LabelEncoder()
    # encode_Y_train = pd.Series(label_encoder.fit_transform(Y_train), index=Y_train.index)
    # encode_Y_test = pd.Series(label_encoder.fit_transform(Y_test), index=Y_test.index)
    # encode_Y_val = pd.Series(label_encoder.fit_transform(Y_val), index=Y_val.index)
    # train_classes = len(set(encode_Y_train))
    # test_classes = len(set(encode_Y_test))
    # val_classes = len(set(encode_Y_val))
    # print(f"Number of train labels: {train_classes}")
    # print(f"Number of test labels: {test_classes}")
    # print(f"Number of val labels: {val_classes}")

    
    # 二分类标签映射
    # label_map = {'DoS Slowhttptest': 1, 'BENIGN': 0, 'DoS GoldenEye': 1, 'Bot': 1, 'DoS Hulk': 1, 'Web Attack � Brute Force': 1, 'FTP-Patator': 1, 'DoS slowloris': 1, 'SSH-Patator': 1}
    # bin_y_train = Y_train.map(label_map)
    # bin_y_val = Y_val.map(label_map)
    # bin_y_test = Y_test.map(label_map)

    # # 多分类标签映射 尝试
    # label_map = {'DoS Slowhttptest': 1, 'BENIGN': 0, 'DoS GoldenEye': 1, 'Bot': 2, 'DoS Hulk': 1, 'Web Attack � Brute Force': 3, 'FTP-Patator': 4, 'DoS slowloris': 1, 'SSH-Patator': 4}
    # multi_y_train = Y_train.map(label_map)
    # multi_y_val = Y_val.map(label_map)
    # multi_y_test = Y_test.map(label_map)

    print("原始标签 classes_：", Y_train.unique())
    print("标签 → 编码 映射：")
    for i, label in enumerate(Y_train.unique()):
        print(f"{label} → {i}")

    # sys.exit()


    # if val_classes != train_classes or test_classes!= train_classes:
    #     sys.exit()

####### 增加列实验 ###############
    # correlation = X_train.corrwith(encode_Y_train)
    # print("len:",len(correlation))
    # high_corr_features = correlation[abs(correlation) > 0.07].index
    # remaining = X_train.columns.drop(high_corr_features)
    # low_corr_features = correlation[abs(correlation) <= 0.09].index
    # df_high_corr = X_train[high_corr_features]    # 高相关性的特征
    # df_low_corr = X_train[low_corr_features]      # 低相关性的特征

    # df_high_val = X_val[high_corr_features]    # 高相关性的特征
    # df_low_val = X_val[low_corr_features]      # 低相关性的特征

    # df_high_test = X_test[high_corr_features]    # 高相关性的特征
    # df_low_test = X_test[low_corr_features]      # 低相关性的特征

    # # 取前五个特征
    # # top_high_corr = high_corr_features[:28]
    # top_low_corr = remaining[:33]

    # # 拼接高相关性和低相关性的前五个特征
    # combined_features = pd.Index(high_corr_features).append(pd.Index(top_low_corr))
    # # 混合特征
    # combined_test = X_test[combined_features] 


    # print("high_corr_features:",len(high_corr_features))
    # print("low_corr_features:",len(low_corr_features))
    # print('remaining:',len(remaining))
    # # sys.exit()




    # encode_Y_train = label_encoder.fit_transform(Y_train)
    # encode_Y_test = label_encoder.fit_transform(Y_test)
    # encode_Y_val = label_encoder.fit_transform(Y_val)
    # print(type(encode_Y_train))
    # print(type(Y_train))
    #print(encode_Y_train.dtype)
    # input_Y_train=pd.DataFrame(encode_Y_train)
    # input_Y_test=pd.dataframe(encode_Y_test)
    # input_Y_val=pd.dataframe(encode_Y_val)
    # print(type(input_Y_train))

    #sys.exit()

    print("train:",len(X_train)/len(cic2017.index)*100,"%")
    print("test:",len(X_test)/len(cic2017.index)*100,"%")
    print("val:",len(X_val)/len(cic2017.index)*100,"%")

    
    # load dataset by specifying dataset name
    #allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_NSLKDD()
    
    # 消融实验
    if os.path.exists('/home/zhangheming/code/code/feature_type/cicids2017/numerical_feature_2017.txt'):
        with open('/home/zhangheming/code/code/feature_type/cicids2017/numerical_feature_2017.txt', 'r') as f:
            num_cols = [x.strip().lower() for x in f.readlines()]
    else:
        num_cols = []
    if os.path.exists('binary_feature.txt'):
        with open('binary_feature.txt', 'r') as f:
            bin_cols = [x.strip().lower() for x in f.readlines()]
    else:
        bin_cols = []
    cat_cols = [col for col in X_train.columns if col not in num_cols and col not in bin_cols]
    
    # cat_cols = []   #
    # num_cols = X_train.columns
    # bin_cols = []


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build classifier
    #model=ids.train()
 
    # model = ids.build_classifier(cat_cols, num_cols, bin_cols, num_class=7)    ###
    # model.to(device)
    # print(model.device)


    # gdf = cudf.DataFrame.from_pandas(df)

    # g_X_train = cudf.DataFrame.from_pandas(X_train.values)
    # g_encode_Y_train = cudf.DataFrame.from_pandas(encode_Y_train.values)
    # g_X_val = cudf.DataFrame.from_pandas(X_val.values)
    # g_encode_Y_val = cudf.DataFrame.from_pandas(encode_Y_val.values)

    # 正常实验训练
    # print("标签分布:", encode_Y_train.value_counts()+encode_Y_val.value_counts()+encode_Y_test.value_counts())
    # # # sys.exit()

    # sys.exit()
    model_bin = ids.build_classifier(cat_cols, num_cols, bin_cols, num_class=2)    ###
    model_bin.to(device)
    print(model_bin.device)
    # 二分类实验
    ids.train(model_bin, (X_train, Y_train), (X_val, Y_val), lr=1e-4, num_epoch = 50, eval_metric = 'acc', output_dir = './bin_2017')      # 学习率要1e-3 10epoch  99.31%
    # 二分类实验测试
    test_model(cat_cols, num_cols, bin_cols, X_test, Y_test, './bin_2017', class_num = 2)

    # model_mul = ids.build_classifier(cat_cols, num_cols, bin_cols, num_class=5)    ###
    # model_mul.to(device)
    # print(model.device)
    # # 多分类实验
    # ids.train(model_mul, (X_train, multi_y_train), (X_val, multi_y_val), lr=1e-4, num_epoch = 20, eval_metric = 'acc', output_dir = './multi_2017')      # 学习率要1e-3 10epoch  99.31%
    # # 多分类实验测试
    # test_model(cat_cols, num_cols, bin_cols, X_test, multi_y_test, './multi_2017', class_num = 5)  

    # model_more = ids.build_classifier(cat_cols, num_cols, bin_cols, num_class=7)    ###
    # model_more.to(device)
    # print(model.device)
    # # 多分类实验（更细粒度）
    # ids.train(model_more, (X_train, encode_Y_train), (X_val, encode_Y_val), lr=1e-4, num_epoch = 20, eval_metric = 'acc', output_dir = './multi_cat_more_2017')      # 学习率要1e-3 10epoch  99.31%
    # # 多分类实验测试（更细粒度）
    # test_model(cat_cols, num_cols, bin_cols, X_test, encode_Y_test, './multi_cat_more_2017', class_num = 7)  


    # # 增列实验训练
    # ids.train(model, (df_high_corr, encode_Y_train), (df_high_val, encode_Y_val), lr = 1e-4, num_epoch = 10, eval_metric = 'conf_mat', output_dir = './ckpt_40')      # 学习率要1e-3 10epoch  99.31%
    # # 增列实验测试
    # test_model(cat_cols, num_cols, bin_cols, combined_test, encode_Y_test, class_num = 7, model_path= './ckpt_40')  

    
    # SHAP
    # 2. 准备SHAP分析数据  
    # x_test, y_test = testset  
    x_background = X_train.sample(10, random_state=42)  
    x_explain = X_test.sample(10, random_state=42)  
    x_explain = x_explain.fillna(0)


    def predict_for_shap(x_df):
        model_bin.eval()
        x_tensor = torch.tensor(x_df.values, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model_bin(x_df)  # x_df 是 pd.DataFrame
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits).cpu().numpy()
            # print("模型输出：", logits)
            # print("模型输出：", probs)
        return probs

    
    # 4. 创建SHAP解释器  
    explainer = shap.Explainer(predict_for_shap, x_background, algorithm="permutation", model_output="logit", batch_size=1)
    # explainer = shap.Explainer(predict_for_shap, x_background)  
    shap_values = explainer(x_explain)  

    

    
    # 5. 生成可视化  
    shap.summary_plot(shap_values.values, x_explain, show=False)
    plt.title('Feature Importance Summary')
    plt.savefig("shap_summary_plot.png")
    plt.clf()  # 清空当前图

    # 2. 条形图
    shap.summary_plot(shap_values.values, x_explain, plot_type="bar", show=False)
    plt.title('Mean Feature Importance')
    plt.savefig("shap_bar_plot.png")
    plt.clf()

    # 3. 瀑布图（第一个样本）
    shap.waterfall_plot(shap_values[0], show=False)
    plt.title('Waterfall Plot for Sample 0')
    plt.savefig("shap_waterfall_plot.png")
    plt.clf()
    print(shap_values[0].values)   # 所有特征的 SHAP 值（对预测的影响）
    print(shap_values[0].data)     # 这条样本中各特征的原始值
    print(shap_values[0].base_values)  # 模型输出的基线值（没有任何特征时的预测）

    # shap.save_html("shap_force_plot.html", force_plot_html)
    html = shap.force_plot(
        shap_values.base_values[0],
        shap_values.values[0],
        x_explain.iloc[0],
        matplotlib=False
    )
    with open("shap_force_plot_sample0.html", "w") as f:
        f.write(shap.getjs())
        f.write(html.html())
    
    

