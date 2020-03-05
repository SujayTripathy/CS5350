from __future__ import division
import pandas as pd
import numpy as np
import csv

def wTraining(data,epoch,rate,weights):
    for t in range(epoch):
        for i in range(len(data)):
            prediction=predict(data[i][:-1],weights)
            error=data[i][-1]-prediction
            for j in range(len(weights)):
                weights[j]=weights[j]+(rate*error*data[i][j])
    return weights


def predict(inputs,weights):
    total=0
    for input,weight in zip(inputs,weights):
        total+=input*weight
    return 1 if total>0 else 0

def testing(test,weights):
    correct=0
    predictions=[]
    for i in range(len(test)):
        prediction=predict(test[i][:-1],weights)
        predictions.append(prediction)
        if prediction==test[i][-1]: correct+=1
    print correct/len(test)

def votedTraining(data,epoch,rate,weights,m):
    allw=[]
    count=[]
    allw.append(weights)
    countM=0
    count.append(countM)
    for t in range(epoch):
        for i in range(len(data)):
            prediction=predict(data[i][:-1],allw[m])
            error=data[i][-1]-prediction
            if error!=0:   
                allw.append([0,0,0,0])       
                for j in range(len(weights)):
                    allw[m+1][j]=allw[m][j]+(rate*error*data[i][j])
                m=m+1
                count.append(countM)
                countM=1
            else:
                countM=countM+1
    return allw, count


def votedTesting(test,weights,counts):
    correct=0
    for i in range(len(test)):
        prediction=0
        prediction+=counts[i]*predict(test[i][:-1],weights[i])
        if prediction>0:
            prediction=1
        else:
            prediction=0
        if prediction==test[i][-1]: correct+=1
    print correct/len(test)

def averageTraining(data,epoch,rate,weights,a):
    for t in range(epoch):
        for i in range(len(data)):
            prediction=predict(data[i][:-1],weights)
            error=data[i][-1]-prediction
            for j in range(len(weights)):
                weights[j]=weights[j]+(rate*error*data[i][j])
                a[j]=a[j]+weights[j]
    return a


def main():
    training=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')
    weights=[0,0,0,0]
    learningRate=0.1
    epoch=10
    m=0
    a=[0,0,0,0]
    trainedWeight=wTraining(training.values,epoch,learningRate,weights)
    print "Standard trained weight is:"
    print trainedWeight
    print "Accuracy of standard perceptron is:"
    testing(test.values,trainedWeight)
    weights=[0,0,0,0]
    votedWeights,count=votedTraining(training.values,epoch,0.1,weights,m)
    print "Number of weights are: "
    print len(votedWeights)
    print "Count of voted weights is: "
    print count
    print "Accuracy of voted perceptron is:"
    votedTesting(test.values,votedWeights,count)
    weights=[0,0,0,0]
    averageWeight=averageTraining(training.values,epoch,learningRate,weights,a)
    print "Average weight vector is:"
    print averageWeight
    print "Accuracy of average perceptron is:"
    testing(test.values,averageWeight)




if __name__== '__main__':
    main()
