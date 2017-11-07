import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    train_data['Age'] = train_data['Age'].apply(lambda x:int(x/10))
    test_data['Age'] = test_data['Age'].apply(lambda x:int(x/10))
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


def get_split_feature_entropy(fname,data_set):
    data_num = data_set.shape[0]
    feature_entropy = {}
    color={0:'r',1:'g'}
    for feature,feature_num in dict(data_set.groupby(fname).size()).items():
        label_set = dict(data_set[data_set[fname]==feature].groupby("Survived").size())
        entropy_tmp = 0
        width = 0.3
        i = -0.15
        for key, val in label_set.items():
            prob = val / feature_num
            entropy_tmp -= feature_num/data_num * prob * math.log(prob, 2)
            plt.bar(feature+i, val, width=width,label=key,color=color[key])
            i=i+0.3

        feature_entropy[feature]=entropy_tmp
    plt.xticks(list(dict(data_set.groupby(fname).size()).keys()))
    plt.title(fname)
    # plt.show()
    result = 0
    for val in feature_entropy.values():
        result+=val
    return result


def generate_tree(former_tree_node,train_data):
    if train_data.shape[1]==1:
        former_tree_node['end']=train_data.iloc[0,0]
        return
    data_set = get_split_feature_set(train_data)
    entropy_total = get_entropy(trainData)
    # print(entropy_total)
    entropy_dict = {}
    for key in data_set:
        data = (data_set[key])
        entropy_dict[key] = entropy_total - (get_split_feature_entropy(key, data))
    best = max(entropy_dict, key=entropy_dict.get)
    former_tree_node[best]={}
    feat_list=list(train_data.columns)
    feat_list.remove(best)
    for i in dict(train_data.groupby(best).size()).keys():
        former_tree_node[best][i]={}
        generate_tree(former_tree_node[best][i],train_data[feat_list][train_data[best] == i])


def get_end(data_row,tree):
    key = list(tree.keys())[0]
    res = tree[key][data_row[key]]
    if 'end' not in list(res.keys()):
        return get_end(data_row,res)
    else:
        return res['end']



rawTrainData = pd.DataFrame(pd.read_csv('train.csv'))
rawTestData = pd.DataFrame(pd.read_csv('test.csv'))
idList = np.array(rawTestData.ix[:, 'PassengerId'])
trainData, testData = proc_data(rawTrainData, rawTestData)
# print(trainData)

decisionTree={}
generate_tree(decisionTree,trainData)
print(decisionTree)
count = 0
countF = 0
for _,row in trainData.iterrows():
    result = get_end(row,decisionTree)
    if result!=row['Survived']:
        countF=countF+1
    count = count + 1
print(countF/count)