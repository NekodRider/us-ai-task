import numpy as np
import pandas as pd
import csv
import math

sexDict = {
    'male':0,
    'female':1
}

def proc_data(raw_train_data,raw_test_data):
    # raw_data['Cabin'] = raw_data['Cabin'].fillna('@')
    # raw_data['Cabin'] = raw_data['Cabin'].map(lambda x: ord(x[0])-ord('A'))
    raw_train_data=raw_train_data.assign(Relative=((1-raw_train_data['SibSp'] - raw_train_data['Parch'])**2))
    raw_test_data = raw_test_data.assign(Relative=((1 - raw_test_data['SibSp'] - raw_test_data['Parch']) ** 2))
    train_data = raw_train_data.ix[:, ['Pclass', 'Sex', 'Age', 'SibSp','Parch','Survived']]
    test_data = raw_test_data.ix[:, ['Pclass', 'Sex', 'Age',  'SibSp', 'Parch']]

    data_mean=(raw_test_data.mean()['Age']*raw_test_data.size+raw_train_data.mean()['Age']*raw_train_data.size)/(raw_test_data.size+raw_train_data.size)
    train_data = train_data.fillna({'Age':data_mean })
    test_data = test_data.fillna({'Age': data_mean})


    train_data['Sex'] = train_data['Sex'].map(sexDict)
    test_data['Sex'] = test_data['Sex'].map(sexDict)
    # data['Embarked'] = data['Embarked'].map(embarkedDict)
    return train_data,test_data


def save_result(idList,output):
    result=np.hstack((idList.reshape(idList.size, 1), np.array(output).reshape(idList.size, 1)))
    with open('result.csv','w',newline='') as myFile:
        writer=csv.writer(myFile,delimiter=',')
        writer.writerow(['PassengerId','Survived'])
        for i in list(result):
            writer.writerow(i)


def get_entropy(data_set):
    data_num = data_set.shape[0]
    label_set = {}
    for label,value in dict(data_set.groupby("Survived").size()).items():
        label_set[label]=value
    entropy = 0
    for _,value in label_set.items():
        prob = value/data_num
        entropy -= prob*math.log(prob,2)
    return entropy


def get_split_feature_set(train_data):
    feature_dict={}
    for i in range(train_data.shape[1]-1):
        feature_dict[train_data.columns[i]]=train_data.ix[:,[i,-1]]
    return feature_dict


def get_best(fname,data_set):
    data_num = data_set.shape[0]
    feature_entropy = {}
    feature_prob = {}
    for feature,feature_num in dict(data_set.groupby(fname).size()).items():
        label_set = dict(data_set[data_set[fname]==feature].groupby("Survived").size())
        entropy_tmp = 0
        for key, val in label_set.items():
            prob = val / feature_num
            entropy_tmp -= feature_num/data_num * prob * math.log(prob, 2)
        feature_entropy[feature]=entropy_tmp
    result = 0
    for val in feature_entropy:
        result+=val
    return result


rawTrainData = pd.DataFrame(pd.read_csv('train.csv'))
rawTestData = pd.DataFrame(pd.read_csv('test.csv'))
idList = np.array(rawTestData.ix[:, 'PassengerId'])
trainData, testData = proc_data(rawTrainData, rawTestData)
# print(trainData)
setToCalc = get_split_feature_set(trainData)
entropy_total = get_entropy(trainData)
print(entropy_total)
entropy_dict = {}
for key in setToCalc:
    entropy_dict[key]=(get_best(key,setToCalc[key]))
print(entropy_dict)