from __future__ import division
import numpy as np
import pandas as pd



def LRMAP(T,weight,data,rate,d,v):
    for i in range(T):
        data=data.sample(frac=1).reset_index(drop=True)
        for j in range(data.shape[0]):
            for k in range(len(weight)):
                weight[k]=weight[k]-(rate/(1+(rate/d)*i))*((weight[k]/v)
                -(data.values[j][-1]*data.values[j][k]/(1+np.exp(data.values[j][-1]*data.values[j][k]*weight[k]))))
    return weight


def LRML(T,weight,data,rate,d,v):
    for i in range(T):
        data=data.sample(frac=1).reset_index(drop=True)
        for j in range(data.shape[0]):
            for k in range(len(weight)):
                weight[k]=weight[k]-(rate/(1+(rate/d)*i))*(-1)
                (data.values[j][-1]*data.values[j][k]/(1+np.exp(data.values[j][-1]*data.values[j][k]*weight[k])))
    return weight    


def predict(inputs,weights):
    total=0
    for input,weight in zip(inputs,weights):
        total+=input*weight
    return 1 if total>0 else 0


def test(test,weights):
    correct=0
    predictions=[]
    for i in range(len(test)):
        prediction=predict(test[i][:-1],weights)
        predictions.append(prediction)
        if prediction==test[i][-1]: correct+=1
    print correct/len(test)


def main():
    training=pd.read_csv("train.csv",header=None)
    testing=pd.read_csv("test.csv",header=None)
    weights=[0,0,0]
    epochs=100
    rate=0.01
    d=0.001
    v=[0.01,0.1,0.5,1,3,5,10,100]

    
    print "MAP ACCURACY\n"
    for i in range(len(v)):
        trainedWeight=LRMAP(epochs,weights,training,rate,d,v[i])
        print ("Training accuracy for v="+str(v[i])+" is: ")
        test(training.values,trainedWeight)
        
        print("Testing accuracy for v="+str(v[i])+" is: ")
        test(testing.values,trainedWeight)
        weights=[0,0,0]

    print "MLE ACCURACY\n"
    trainedWeight=LRML(epochs,weights,training,rate,d,0.01)
    print ("Training accuracy for v="+str(v[0])+"is: ")
    test(training.values,trainedWeight)
    
    print("Testing accuracy for v="+str(v[0])+"is: ")
    test(testing.values,trainedWeight)
















if __name__=="__main__":
    main()