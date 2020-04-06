from __future__ import division
import pandas as pd
import numpy as np


def SSG(data,weights,epochs,C,rate,d):
    for t in range(epochs):
        data=data.sample(frac=1).reset_index(drop=True)                                 ##shuffling data
        for i in range(len(data)):
            prediction=predict(data.loc[i],data.loc[i][-1],weights)
            if prediction <= 1:
                for j in range (len(weights)):
                    weights[j]=weights[j]+((rate/(1+(rate/d)*i))*C*len(data)*data.loc[i][-1]*data.loc[i][j])
    return weights

def SSG2(data,weights,epochs,C,rate):
    for t in range(epochs):
        data=data.sample(frac=1).reset_index(drop=True)                                 ##shuffling data
        for i in range(len(data)):
            prediction=predict(data.loc[i],data.loc[i][-1],weights)
            if prediction <= 1:
                for j in range (len(weights)):
                    weights[j]=weights[j]+((rate/(1+i))*C*len(data)*data.loc[i][-1]*data.loc[i][j])
    return weights


def predict(inputs,label,weights):
    total=0
    for input,weight in zip(inputs,weights):
        total+=input*weight
    total=total*label
    return total    

def test(test,weights):
    correct=0
    for i in range(len(test)):
        prediction=predict(test.loc[i],test.loc[i][-1],weights)
        if prediction>=0: prediction=1
        else:
            prediction=0
        if prediction==test.loc[i][-1]: correct+=1
    print correct/len(test)    
           

def main():
    training=pd.read_csv('train.csv',names=['A','B','C','D','label'])
    testing=pd.read_csv('test.csv',names=['A','B','C','D','label'])
    training.loc[training['label']==0,'label']=-1                                       ##Changing 0 to -1 in label
    testing.loc[testing['label']==0,'label']=-1
    weights=[0,0,0,0]
    epochs=100
    C=[100/873,500/873,700/873]
    rates=[0.01,0.005,0.0025]
    d=8
    print "First Rate Formula"
    weights=SSG(training,weights,epochs,C[0],0.0001,32)
    print weights
    print "Training error is: "
    test(training,weights)
    print "Testing error is: "
    test(testing,weights)
    weights=[0,0,0,0]
    weights=SSG(training,weights,epochs,C[1],0.0001,32)
    print weights
    print "Training error is: "
    test(training,weights)
    print "Testing error is: "
    test(testing,weights)
    weights=[0,0,0,0]
    weights=SSG(training,weights,epochs,C[2],0.0001,32)
    print weights
    print "Training error is: "
    test(training,weights)
    print "Testing error is: "
    test(testing,weights)


    print "2nd Rate Formula"
    weights=SSG2(training,weights,epochs,C[0],2)
    print weights
    print "Training error is: "
    test(training,weights)
    print "Testing error is: "
    test(testing,weights)
    weights=[0,0,0,0]
    weights=SSG2(training,weights,epochs,C[1],1.10)
    print weights
    print "Training error is: "
    test(training,weights)
    print "Testing error is: "
    test(testing,weights)
    weights=[0,0,0,0]
    weights=SSG2(training,weights,epochs,C[2],1.05)
    print weights
    print "Training error is: "
    test(training,weights)
    print "Testing error is: "
    test(testing,weights)




if __name__=='__main__':
    main()
