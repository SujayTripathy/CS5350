import numpy as np
import pandas as pd
import sys
from pprint import pprint





data=pd.read_csv('train.csv',names=['buying','maint','doors',
'persons','lug_boot','safety','label'])

def Entropy(feature):
    values,count=np.unique(feature,return_counts=True)
    entropy=np.sum([(-count[i]/np.sum(count))*np.log2(count[i]/np.sum(count)) for i in range(len(values))])
    return entropy

def GiniIndex(feature):
    values,count=np.unique(feature,return_counts=True)
    giniIndex=1-np.sum(np.square([count[i]/np.sum(count) for i in range(len(values))]))
    return giniIndex

def MajorityError(feature):
    values,count=np.unique(feature,return_counts=True)
    majorityError=np.min(count)/np.sum(count)
    return majorityError

def InformationGain(data,feature,heuristic,label="label"):
    information=heuristic(data[label])
    values,count=np.unique(data[feature],return_counts=True)
    weightedInfo=np.sum([(count[i]/np.sum(count))*heuristic(np.where(data[feature]==values[i],data)[label])for i in range(len(values))])
    gain=information-weightedInfo
    return gain

def ID3(S,Scopy,features,heuristic,label="label",parent_node=None):
    if len(np.unique(data[label]))<=1:
        return np.unique(data[label])[0]
    elif len(data)==0:
        return np.unique(Scopy[label])[np.argmax(np.unique(Scopy[label],return_counts=True)[1])]
    elif len(features)==0:
        return parent_node
    else:
        parent_node=np.unique(data[label])[np.argmax(np.unique(data[label],return_counts=True)[1])]
        gains=[heuristic(data,feature,heuristic,label)for feature in features]
        bestfeature_index=np.argmax(gains)
        bestfeature=features[bestfeature_index]
        tree={bestfeature:{}}
        features=[i for i in features if i != bestfeature]
        for value in np.unique(data[bestfeature]):
            value=value
            splitdata=np.where(data[bestfeature]==value,data)
            subtree=ID3(splitdata,data,features,label,parent_node)
            tree[bestfeature][value]=subtree
    return(tree)
    


if __name__ == "__main__":
    if(len(sys.argv)!=2):
        print "Please enter Information gain method and tree depth."
        sys.exit()
    else:
        if(sys.argv==0):
            ID3()
        elif(sys.argv==1):
            ID3()
        elif(sys.argv==2):
            ID3()
        else:
            print "Invalid arguments"
            sys.exit()