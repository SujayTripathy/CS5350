import numpy as np
import pandas as pd
import argparse
import matplotlib


def cost(X,y,weight):
    cost=0
    print X
    for i in range(len(y)):
        cost+=(y[i]-np.dot(X,weight))**2
    cost/=2
    return cost




if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Batch LMS for concrete data")
    trainingData=pd.read_csv('train.csv',names=['cement','slag','flyash','water','SP','coarseaggr','fineagger','output'
        ])
    testingData=pd.read_csv('test.csv',names=['cement','slag','flyash','water','SP','coarseaggr','fineagger','output'
        ])    

r=1
tolerancelevel=0.000001
weight=np.array([0,0,0,0,0,0,0])

print cost(trainingData.drop("output",1),trainingData["output"],weight)


