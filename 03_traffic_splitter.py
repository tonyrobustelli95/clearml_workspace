# the following script splits each file provided by 02_traffic_manager.py according to a specific criteria (e.g. 70/30, 80/20)

import pandas as pd
import os
import numpy as np

"""
loads each .csv file as dataframe and split it according to the splitting criteria
Then, it stores the whole train and test dataframes in './dataset/split'

trainPercentage: percentage of training data samples. 
If trainPercentage = 100 the function stores only one file representing the whole dataframe
categoryList: list of traffic categories (e.g. ['Browsing','Chat'])
"""
def datasetSplitter(trainPercentage,categoryList):

    dfListTrain = []
    dfListTest = []

    print("Splitting criteria used: " + str(trainPercentage) + "/" + str(100 - trainPercentage) + '\n')

    for i in range(len(catList)):

        df = pd.read_csv('./dataset/scaled/' + catList[i])
        if df.__len__() > 0: 
            
            cat = str(catList[i]).replace(".csv","")
            print(cat + '\n')

            trainLen = int((trainPercentage * df.__len__()) / 100)
            dfSplit = np.split(df.to_numpy(),[trainLen])

            print("Trainig samples: " + str(dfSplit[0].__len__()))
            print("Testing samples: " + str(dfSplit[1].__len__()))
            print("\n")

            train = pd.DataFrame(dfSplit[0],columns=df.columns)
            labelTrain = np.zeros(train.__len__()).astype(int) + i
            train['category'] = labelTrain
            dfListTrain.append(train)

            test = pd.DataFrame(dfSplit[1],columns=df.columns)
            labelTest = np.zeros(test.__len__()).astype(int) + i 
            test['category'] = labelTest
            dfListTest.append(test)
    
    dftrain = pd.concat(dfListTrain,axis=0)

    if trainPercentage == 100:
        dftrain.to_csv('./dataset/split/wholedataset.csv',index=False)
    else:
        dftrain.to_csv('./dataset/split/' + 'train' + str(trainPercentage) + str(100 - trainPercentage) + ".csv",index=False)

        dftest = pd.concat(dfListTest,axis=0)
        dftest.to_csv('./dataset/split/' + 'test' + str(trainPercentage) + str(100 - trainPercentage) + ".csv",index=False)

if __name__ == '__main__':

    catList = sorted(os.listdir('./dataset/scaled/'))

    datasetSplitter(trainPercentage=70,categoryList=catList)
    datasetSplitter(trainPercentage=80,categoryList=catList)
    datasetSplitter(trainPercentage=100,categoryList=catList)