from __future__ import division
import pandas as pd
import numpy as np
import argparse
import math
from scipy import stats
from pprint import pprint

def Entropy(feature,D):
    values,count = np.unique(feature,return_counts = True)
    entropy = np.sum([(-count[i]*D[i]/np.sum(count))*np.log2(count[i]*D[i]/np.sum(count)) for i in range(len(values))])
    return entropy

def InformationGain(data,feature,heuristic,D,label="y"):
    totalEntropy = heuristic(data[label],D)
    values,count= np.unique(data[feature],return_counts=True)
    weightedEntropy = np.sum([(count[i]*D[i]/np.sum(count))*heuristic(data.where(data[feature]==values[i]).dropna()[label],D) for i in range(len(values))])
    informationGain = totalEntropy-weightedEntropy
    return informationGain

# def ID3(S,Scopy,features,heuristic,D,label="y",parentNode = None):
#     if len(np.unique(S[label])) <= 1:
#         return np.unique(S[label])[0]
#     elif len(S)==0:
#         return np.unique(Scopy[label])[np.argmax(np.unique(Scopy[label],return_counts=True)[1])]
#     elif len(features) ==0:
#         return parentNode  
#     else:       
#         parentNode = np.unique(S[label])[np.argmax(np.unique(S[label],return_counts=True)[1])]    
#         item_values = [InformationGain(S,feature,heuristic,D,label) for feature in features] 
#         best_feature_index = np.argmax(item_values)
#         best_feature = features[best_feature_index]
#         tree = {best_feature:{}} 
#         features = [i for i in features if i != best_feature]
#         if len(features)+1 < len(Scopy.columns)-1:
#             return (tree)
#         for value in np.unique(S[best_feature]):
#             value = value
#             sub_data = S.where(S[best_feature] == value).dropna()           
#             subtree = ID3(sub_data,trainingData,features,heuristic,D,label,parentNode)           
#             tree[best_feature][value] = subtree 
#             print subtree          
#         return(tree)

def ID2(S,features,D,label="y"):
    if len(np.unique(S[label]))<=1:
        return np.unique(S[label])[0]
    gains=[InformationGain(S,feature,Entropy,D) for feature in features]
    bestfeatureindex=np.argmax(gains)
    bestfeature=features[bestfeatureindex]
    tree={bestfeature:{}}
    features=[i for i in features if i!=bestfeature]
    for value in np.unique(S[bestfeature]):
        sub_data=S.where(S[bestfeature]==value).dropna()
        tree[bestfeature][value]=sub_data["y"].mode()[0]
    return tree



def Adaboost(T,S):
    D=np.full(np.size(S,0),1/np.size(S,0))
    queries=S.iloc[:,:-1].to_dict(orient="records")
    FinalAlpha=np.full(T,0,np.object_)
    FinalH=np.full(T,0,np.object_)
    for i in range(T):
        tree=ID2(S,S.columns[:-1],D)
        predictions=np.full(np.size(S,0),0,np.object_)
        error=0
        for j in range(len(D)):
            predictions[j]=predict(queries[j],tree)                  ##Gets all the predictions
        for j in range(len(D)):
            if predictions[j]!=S["y"][j]:                            ##Checks for predictions and adds the error
                error+=D[j]           
        alpha=0.5*np.log((1-error)/error)
        FinalAlpha[i]=alpha
        FinalH[i]=tree
        for j in range(len(D)):
            if predictions[j]==S["y"][j]:
                D[j]=D[j]*math.exp(-error*1)
            else:
                D[j]=D[j]*math.exp(-error*(-1))
        normfactor=np.sum(D)
        for j in range(len(D)):
            D[j]=D[j]/normfactor
    return FinalAlpha,FinalH
        

def predict(query,tree):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            result=tree[key][query[key]]
            return result
def AdaPredict(query,FinalAlpha,FinalH):
    prediction=0
    for i in range(len(FinalAlpha)):
        result=predict(query,FinalH[i])
        prediction+=result*FinalAlpha[i]
    return np.sign(prediction)




def test(data,alpha,H):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = AdaPredict(queries[i],alpha,H)  
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["y"])/len(data))*100,'%')
    
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Adaboost algorithm for Bank data")

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

    for i in range(len(trainingData)):
        if(trainingData["y"][i]=="yes"):
            trainingData["y"][i]=1
        else:
            trainingData["y"][i]=-1
    
    alpha,H=Adaboost(100,trainingData)
    print alpha,H
    #test(trainingData,alpha,H)
    