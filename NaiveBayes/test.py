import NaiveBayes
import numpy

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 表示带敏感词汇
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:  # 遍历文档列表
        # 首先将当前文档的单词唯一化，然后以交集的方式加入到保存词汇的集合中。
        vocabSet = vocabSet | set(document)

    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("单词: %s不在词汇表当中" % word)
    return returnVec


def test():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)

    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    pV, pAb = NaiveBayes.NaiveBayes_train(numpy.array(trainMat), numpy.array(listClasses))

    # 测试一
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = numpy.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, '分类结果: ', NaiveBayes.NaiveBayes_classify(thisDoc, pV, pAb))

    # 测试二
    testEntry = ['stupid', 'garbage']
    thisDoc = numpy.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, '分类结果: ', NaiveBayes.NaiveBayes_classify(thisDoc, pV, pAb))


if __name__ == '__main__':
    test()
