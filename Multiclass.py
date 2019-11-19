# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:50:34 2019

@author: Lenovo
"""

import numpy as np
import csv
from sklearn.impute import SimpleImputer
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def init_data(dataDir):
    data = []
    with open(dataDir) as csvfile:
        csv_reader = csv.reader(csvfile)  
        #header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  
            data.append(row)

    #统计数据集的行数和列数
    row = 0
    colum = 0
    for i in data:
        row += 1
    for j in data[0]:
        colum += 1
     
    #将字符串？转化为缺失值np.nan
    for i in range(row):
        for j in range(colum):
            if(data[i][j] =='?'):
                data[i][j] = np.nan
     
    #将string型数据转化为float型，并把数据集转化为数组
    data = [[float(x) for x in row] for row in data]
    data = np.array(data)
    
    return data

def Compdefault(data):
    #用属性列的均值填补缺失值
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

    imp_mean.fit(data)
    newData = imp_mean.transform(data)
    
    
    return newData

def Knn(trainData,classLabels,test_data):
    X_train,X_test,y_train,y_test = model_selection.train_test_split(trainData,classLabels,
                                                                     test_size=0.2,random_state=5)
    
    ks = [i for i in range(10,20)]
    
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors = k,weights='uniform')
        clf.fit(trainData,classLabels)
        p = clf.predict(X_test)
        print(k,':',accuracy_score(y_test,p)) 
        
    clf = neighbors.KNeighborsClassifier(n_neighbors = 11,weights='uniform')
    clf.fit(trainData,classLabels)
    test_pred = clf.predict(test_data)
    
    m = len(test_pred)
    prediction = [[] for i in range(m)]
    for i in range(m):  
        prediction[i].append(i+1)
        prediction[i].append(test_pred[i])
        
    # =============================================================================
#     #折线图
#     arrs = []
#     #推导式选K值
#     ks = [i for i in range(90,100)]
#     kz = KFold(n_splits = 5,shuffle = False)
#     for k in ks:
#         arr = []
#         for train_indexs,test_indexs in kz.split(X_train):
#             clf = KNeighborsClassifier(n_neighbors = k)
#             clf.fit(X_train[train_indexs],y_train[train_indexs])
#             p = clf.predict(X_train[test_indexs])
#             arr.append(1. - accuracy_score(y_train[test_indexs],p))
#         arrs.append(arr)
#     arrs = np.array(arrs)
#     mean_arrs = np.mean(arrs,axis = 1)
#     std_arrs = np.std(arrs,axis = 1)
#     plt.figure(figsize = (10,5))
#     plot_data = np.array([mean_arrs,mean_arrs+std_arrs,mean_arrs-std_arrs]).T
#     plt.plot(ks,plot_data,'.-')
# =============================================================================

    return prediction
    


def main():
    train_data = init_data('train.csv')
    test_data = init_data('test.csv')
                                                                                                                                                                                                        

    trainData = train_data[:,:-1]
    classLabels = train_data[:,-1]
    classLabels = classLabels.ravel()


    trainData = Compdefault(trainData)
    test_data = Compdefault(test_data)
    
    
    trainData[:,2] = trainData[:,2] * 2
    trainData[:,3] = trainData[:,3] * 2
    trainData[:,4] = trainData[:,4] * 2
    trainData[:,6] = trainData[:,6] * 2
    trainData[:,11] = trainData[:,11] * 2
    trainData[:,12] = trainData[:,12] * 3
    
    test_data[:,2] = test_data[:,2] * 2
    test_data[:,3] = test_data[:,3] * 2
    test_data[:,4] = test_data[:,4] * 2
    test_data[:,6] = test_data[:,6] * 2
    test_data[:,11] = test_data[:,11] * 2
    test_data[:,12] = test_data[:,12] * 3
    
# =============================================================================
#     trainData[:,2] = trainData[:,2] * 2.5
#     trainData[:,3] = trainData[:,3] * 1.5
#     trainData[:,4] = trainData[:,4] * 2
#     trainData[:,6] = trainData[:,6] * 3
#     trainData[:,11] = trainData[:,11] * 3
#     trainData[:,12] = trainData[:,12] * 0.5
# =============================================================================
     
    prediction = Knn(trainData,classLabels,test_data)
    
    
    return prediction

prediction = main()
f = open(r'C:\Users\Lenovo\Desktop\\sample.csv','w')
np.savetxt(r'C:\Users\Lenovo\Desktop\\sample.csv',prediction,delimiter=',')
f.close()


