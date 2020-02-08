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

def InformationGain(data,feature,heuristic,label="y"):
    totalEntropy = heuristic(data[label])
    values,count= np.unique(data[feature],return_counts=True)
    weightedEntropy = np.sum([(count[i]/np.sum(count))*heuristic(data.where(data[feature]==values[i]).dropna()[label]) for i in range(len(values))])
    informationGain = totalEntropy-weightedEntropy
    return informationGain

def ID3(S,Scopy,features,heuristic,label="y",parentNode = None):
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
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["y"])/len(data))*100,'%')
    
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="ID3 Algorithm with entropy, gini and majority error.")
    parser.add_argument("depth",type=int,help="Max depth for trees")
    args=parser.parse_args()
    maxdepth=args.depth

    trainingData=pd.read_csv('banktrain.csv',names=[
        'age','job','marital','education','default','balance','housing','load','contact',
        'day','month','duration','campaign','pdays','previous','poutcome','y',])
    testingData=pd.read_csv('banktest.csv',names=[
        'age','job','marital','education','default','balance','housing','load','contact',
        'day','month','duration','campaign','pdays','previous','poutcome','y',])
    
    trainingAgeMedian=trainingData['age'].median()
    trainingData.loc[trainingData.age>=trainingAgeMedian,'age']=10000
    trainingData.loc[trainingData.age<trainingAgeMedian,'age']=0
    
    trainingBalanceMedian=trainingData['balance'].median()
    trainingData.loc[trainingData.balance>=trainingBalanceMedian,'balance']=10000000000
    trainingData.loc[trainingData.balance<trainingBalanceMedian,'balance']=0
    
    trainingDayMedian=trainingData['day'].median()
    trainingData.loc[trainingData.day>=trainingDayMedian,'day']=10000000
    trainingData.loc[trainingData.day<trainingDayMedian,'day']=0
    
    trainingDurationMedian=trainingData['duration'].median()
    trainingData.loc[trainingData.duration>=trainingDurationMedian,'duration']=10000000
    trainingData.loc[trainingData.duration<trainingDurationMedian,'duration']=0
    
    trainingCampaignMedian=trainingData['campaign'].median()
    trainingData.loc[trainingData.campaign>=trainingCampaignMedian,'campaign']=10000000
    trainingData.loc[trainingData.campaign<trainingCampaignMedian,'campaign']=0
    
    trainingPdaysMedian=trainingData['pdays'].median()
    trainingData.loc[trainingData.pdays>=trainingPdaysMedian,'pdays']=10000000
    trainingData.loc[trainingData.pdays<trainingPdaysMedian,'pdays']=0
    
    trainingPreviousMedian=trainingData['previous'].median()
    trainingData.loc[trainingData.previous>=trainingPreviousMedian,'previous']=10000000
    trainingData.loc[trainingData.previous<trainingPreviousMedian,'previous']=0

    testingAgeMedian=testingData['age'].median()
    testingData.loc[testingData.age>=testingAgeMedian,'age']=10000
    testingData.loc[testingData.age<testingAgeMedian,'age']=0
    
    testingBalanceMedian=testingData['balance'].median()
    testingData.loc[testingData.balance>=testingBalanceMedian,'balance']=10000000000
    testingData.loc[testingData.balance<testingBalanceMedian,'balance']=0
    
    testingDayMedian=testingData['day'].median()
    testingData.loc[testingData.day>=testingDayMedian,'day']=10000000
    testingData.loc[testingData.day<testingDayMedian,'day']=0
    
    testingDurationMedian=testingData['duration'].median()
    testingData.loc[testingData.duration>=testingDurationMedian,'duration']=10000000
    testingData.loc[testingData.duration<testingDurationMedian,'duration']=0
    
    testingCampaignMedian=testingData['campaign'].median()
    testingData.loc[testingData.campaign>=testingCampaignMedian,'campaign']=10000000
    testingData.loc[testingData.campaign<testingCampaignMedian,'campaign']=0
    
    testingPdaysMedian=testingData['pdays'].median()
    testingData.loc[testingData.pdays>=testingPdaysMedian,'pdays']=10000000
    testingData.loc[testingData.pdays<testingPdaysMedian,'pdays']=0
    
    testingPreviousMedian=testingData['previous'].median()
    testingData.loc[testingData.previous>=testingPreviousMedian,'previous']=10000000
    testingData.loc[testingData.previous<testingPreviousMedian,'previous']=0

    treeEntropy = ID3(trainingData,trainingData,trainingData.columns[:-1],Entropy)
    treeGini = ID3(trainingData,trainingData,trainingData.columns[:-1],GiniIndex)
    treeMajority = ID3(trainingData,trainingData,trainingData.columns[:-1],MajorityError)
    test(trainingData,treeEntropy)
    test(testingData,treeEntropy)
    test(trainingData,treeGini)
    test(testingData,treeGini)
    test(trainingData,treeMajority)
    test(testingData,treeMajority)

    testingJobMode=testingData['job'].mode()[0]
    testingData.loc[testingData.job=='unknown','job']=testingJobMode

    testingEducationMode=testingData['education'].mode()[0]
    testingData.loc[testingData.education=='unknown','education']=testingEducationMode

    testingContactMode=testingData['contact'].mode()[0]
    testingData.loc[testingData.contact=='unknown','contact']=testingContactMode

    testingPoutcomeMode=testingData['poutcome'].mode()[0]
    testingData.loc[testingData.poutcome=='unknown','poutcome']=testingPoutcomeMode
    
    trainingJobMode=trainingData['job'].mode()[0]
    trainingData.loc[trainingData.job=='unknown','job']=trainingJobMode

    trainingEducationMode=trainingData['education'].mode()[0]
    trainingData.loc[trainingData.education=='unknown','education']=trainingEducationMode

    trainingContactMode=trainingData['contact'].mode()[0]
    trainingData.loc[trainingData.contact=='unknown','contact']=trainingContactMode

    trainingPoutcomeMode=trainingData['poutcome'].mode()[0]
    trainingData.loc[trainingData.poutcome=='unknown','poutcome']=trainingPoutcomeMode


    treeEntropyb = ID3(trainingData,trainingData,trainingData.columns[:-1],Entropy)
    treeGinib = ID3(trainingData,trainingData,trainingData.columns[:-1],GiniIndex)
    treeMajorityb = ID3(trainingData,trainingData,trainingData.columns[:-1],MajorityError)
    test(trainingData,treeEntropyb)
    test(testingData,treeEntropyb)
    test(trainingData,treeGinib)
    test(testingData,treeGinib)
    test(trainingData,treeMajorityb)
    test(testingData,treeMajorityb)

