from __future__ import division
import pandas as pd
import numpy as np
import argparse
from pprint import pprint

def Entropy(feature):
    values,count = np.unique(feature,return_counts = True)
    entropy = np.sum([(-count[i]/np.sum(count))*np.log2(count[i]/np.sum(count)) for i in range(len(values))])
    return entropy

def GiniIndex(feature):
    values,count=np.unique(feature,return_counts=True)
    giniIndex=1-np.sum([np.square(count[i]/np.sum(count)) for i in range(len(values))])
    return giniIndex

def MajorityError(feature):
    values,count=np.unique(feature,return_counts=True)
    majorityError=np.min(count)/np.sum(count)
    return majorityError

def InformationGain(data,feature,heuristic,label="label"):
    totalEntropy = heuristic(data[label])
    values,count= np.unique(data[feature],return_counts=True)
    weightedEntropy = np.sum([(count[i]/np.sum(count))*heuristic(data.where(data[feature]==values[i]).dropna()[label]) for i in range(len(values))])
    informationGain = totalEntropy-weightedEntropy
    return informationGain

def ID3(S,Scopy,features,heuristic,label="label",parentNode = None):
    if len(np.unique(S[label])) <= 1:
        return np.unique(S[label])[0]
    elif len(S)==0:
        return np.unique(Scopy[label])[np.argmax(np.unique(Scopy[label],return_counts=True)[1])]
    elif len(features) ==0:
        return parentNode  
    else:       
        parentNode = np.unique(S[label])[np.argmax(np.unique(S[label],return_counts=True)[1])]    
        item_values = [InformationGain(S,feature,heuristic,label) for feature in features] 
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature:{}} 
        features = [i for i in features if i != best_feature]
        if (maxdepth+len(features)) < (len(Scopy.columns)-1):
            return (tree)
        for value in np.unique(S[best_feature]):
            value = value
            sub_data = S.where(S[best_feature] == value).dropna()           
            subtree = ID3(sub_data,trainingData,features,heuristic,label,parentNode)           
            tree[best_feature][value] = subtree           
        return(tree)    
                    
def predict(query,tree,default = 1):
    
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result

def test(data,tree):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["label"])/len(data))*100,'%')
    
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="ID3 Algorithm with entropy, gini and majority error.")
    parser.add_argument("depth",type=int,help="Max depth for trees")
    args=parser.parse_args()
    maxdepth=args.depth
    trainingData =pd.read_csv('train.csv',
                       names=['buying','maint','doors','persons','lug_boot',
                                                   'safety','label',]) 
    testingData=pd.read_csv('test.csv',names=['buying','maint','doors','persons','lug_boot',
                                                    'safety','label',])
    treeEntropy = ID3(trainingData,trainingData,trainingData.columns[:-1],Entropy)
    treeGini = ID3(trainingData,trainingData,trainingData.columns[:-1],GiniIndex)
    treeMajority = ID3(trainingData,trainingData,trainingData.columns[:-1],MajorityError)
    test(trainingData,treeEntropy)
    test(testingData,treeEntropy)
    test(trainingData,treeGini)
    test(testingData,treeGini)
    test(trainingData,treeMajority)
    test(testingData,treeMajority)
    