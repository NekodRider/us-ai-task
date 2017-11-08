import pandas as pd
import numpy as np
import csv


def process_data(raw_train_data, raw_test_data):
    # raw_data['Cabin'] = raw_data['Cabin'].map(lambda x: ord(x[0])-ord('A'))
    raw_train_data = raw_train_data.assign(Relative=((1 - raw_train_data['SibSp'] - raw_train_data['Parch']) ** 2))
    raw_test_data = raw_test_data.assign(Relative=((1 - raw_test_data['SibSp'] - raw_test_data['Parch']) ** 2))
    train_data = raw_train_data.ix[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
    test_data = raw_test_data.ix[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]

    data_mean = (raw_test_data.mean()['Age'] * raw_test_data.size + raw_train_data.mean()[
        'Age'] * raw_train_data.size) / (raw_test_data.size + raw_train_data.size)
    print(data_mean)
    train_data = train_data.fillna({'Age': data_mean})
    test_data = test_data.fillna({'Age': data_mean})

    train_data['Sex'] = train_data['Sex'].map(sexDict)
    test_data['Sex'] = test_data['Sex'].map(sexDict)
    # data['Embarked'] = data['Embarked'].map(embarkedDict)
    return train_data, test_data


def save_result(id_list, out_put):
    result = np.hstack((id_list.reshape(id_list.size, 1), np.array(out_put).reshape(id_list.size, 1)))
    with open('result.csv', 'w', newline='') as myFile:
        writer = csv.writer(myFile, delimiter=',')
        writer.writerow(['PassengerId', 'Survived'])
        for i in list(result):
            writer.writerow(i)


def gaussian(dist, sigma=10.0):
    weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return weight


def knn_classify(test_data, train_data, label, k):
    data_size = train_data.shape[0]
    diff = (np.tile(test_data, (data_size, 1)) - train_data) ** 2
    dist_mat = np.sum(diff, axis=1)
    distance = dist_mat ** 0.5

    sorted_index = np.argsort(distance)
    class_gauss_value = {}
    gauss_sum = sum([gaussian(x) for x in distance[sorted_index[:k]]])
    for i in range(k):
        dist = distance[sorted_index[i]]
        vote_label = label[sorted_index[i]]
        class_gauss_value[vote_label] = class_gauss_value.get(vote_label, 0) + gaussian(dist) / gauss_sum
    result = None
    max_value = 0
    for key, value in class_gauss_value.items():
        if value > max_value:
            result = key
            max_value = value
    return result


def predict_array(test_array, train_data, label, k):
    result = []
    for _, row in test_array.iterrows():
        row_result = knn_classify(row, train_data, label, k)
        result.append(row_result)
    return result


embarkedDict = {
    'C': 1,
    'Q': 2,
    'S': 3
}
sexDict = {
    'male': 0,
    'female': 1
}

rawTrainData = pd.DataFrame(pd.read_csv('train.csv'))
rawTestData = pd.DataFrame(pd.read_csv('test.csv'))
labels = rawTrainData.ix[:, 'Survived']
idList = np.array(rawTestData.ix[:, 'PassengerId'])
trainData, testData = process_data(rawTrainData, rawTestData)
print(trainData, labels)

output = predict_array(testData, trainData, labels, 5)

save_result(idList, output)
