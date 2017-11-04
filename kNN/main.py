import pandas as pd
import numpy as np
import csv


def proc_data(raw_data):
    # raw_data['Cabin'] = raw_data['Cabin'].fillna('@')
    # raw_data['Cabin'] = raw_data['Cabin'].map(lambda x: ord(x[0])-ord('A'))
    raw_data=raw_data.assign(Relative=((1-raw_data['SibSp'] - raw_data['Parch'])**2))
    data = raw_data.ix[:, ['Pclass', 'Sex', 'Age', 'Fare','SibSp','Parch']]
    data = data.fillna(data.mean()['Age'])
    data['Sex'] = data['Sex'].map(sexDict)
    # data['Embarked'] = data['Embarked'].map(embarkedDict)
    return data


def save_result(idList,output):
    result=np.hstack((idList.reshape(idList.size, 1), np.array(output).reshape(idList.size, 1)))
    with open('result.csv','w',newline='') as myFile:
        writer=csv.writer(myFile,delimiter=',')
        writer.writerow(['PassengerId','Survived'])
        for i in list(result):
            writer.writerow(i)


def gaussian(dist,sigma=10.0):
    weight = np.exp(-dist**2/(2*sigma**2))
    return weight


def knn_classify(test_data, train_data, label, k):

    data_size = train_data.shape[0]
    diff = (np.tile(test_data, (data_size, 1)) - train_data) ** 2
    dist_mat = np.sum(diff, axis=1)
    distance = dist_mat ** 0.5

    sorted_index = np.argsort(distance)
    class_gauss_value = {}
    gauss_sum = sum([ gaussian(x) for x in distance[sorted_index[:k]]])
    for i in range(k):
        dist = distance[sorted_index[i]]
        vote_label = label[sorted_index[i]]
        class_gauss_value[vote_label] = class_gauss_value.get(vote_label, 0) + gaussian(dist)/gauss_sum
    result = None
    max_value = 0
    for key,value in class_gauss_value.items():
        if value > max_value:
            result = key
            max_value = value
    return result


def predict_array(test_array,train_data,label,k):
    result=[]
    for _,row in test_array.iterrows():
        row_result=knn_classify(row, train_data, label, k)
        result.append(row_result)
    return result

embarkedDict={
    'C':1,
    'Q':2,
    'S':3
}
sexDict = {
    'male':0,
    'female':1
}

rawTrainData = pd.DataFrame(pd.read_csv('train.csv'))
rawTestData = pd.DataFrame(pd.read_csv('test.csv'))
labels = rawTrainData.ix[:, 'Survived']
idList = np.array(rawTestData.ix[:, 'PassengerId'])
trainData=proc_data(rawTrainData)
testData=proc_data(rawTestData)
print(trainData,labels)

output = predict_array(testData,trainData,labels,10)

save_result(idList,output)
