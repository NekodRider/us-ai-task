import numpy

def NaiveBayes_train(train_set, train_labels):
    num_train = len(train_set)
    num_feat = len(train_set[0])
    num_labels = max(train_labels) + 1
    pAbusive = [0] * num_labels
    for i in range(num_labels):
        for j in train_labels:
            if j == i:
                pAbusive[i] += 1
    pAbusive = [i / float(num_train) for i in pAbusive]
    #初始化为1和2防止0概率出现
    pNum = [numpy.ones(num_feat) for i in range(num_labels)]
    pDenom = [2.0 for i in range(num_labels)]
    for i in range(num_train):
        for j in range(num_labels):
            if train_labels[i] == j:
                pNum[j] += train_set[i]
                pDenom[j] += sum(train_set[i])
            print(pNum[j])
    pVect = [numpy.log(pNum[i] / pDenom[i]) for i in range(num_labels)]
    return pVect, pAbusive


def NaiveBayes_classify(test_set, pVec, pClass):
    p = [0] * len(pVec)
    for i in range(len(pVec)):
        p[i] = sum(test_set * pVec[i]) + numpy.log(pClass[i])
    return p.index(max(p))