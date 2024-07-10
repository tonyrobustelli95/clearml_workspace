# the following script concatenates csv files related to a specific traffic category

import pandas as pd
import os

if __name__ == '__main__':

    dfList = []
    category = "filetran"

    #loading single .csv files as dataframe and concatenation
    fileList = os.listdir('./dataset/unprocessed/' + category)

    for file in fileList:

        df = pd.read_csv('./dataset/unprocessed/' + category + "/" + file)
        
        if df.__len__() > 0: dfList.append(df)

    df = pd.concat(dfList)

    #delete unnecessary columns
    del df['srcIPClass']
    del df['dstIPClass']
    del df['freqSrc']
    del df['freqDst']

    #print and storage of the concatenated dataframe
    print(df)

    df.to_csv('./dataset/processed/' + category + ".csv",index=False)