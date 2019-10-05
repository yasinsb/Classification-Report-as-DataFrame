"""
Created on Wed Dec 19 13:31:57 2018

@author: Yasin Salimibeni

This small module is created to simply convert a classification report into a 
dataframe for the ease of use and comparision. It will also be easier to use to_csv
method to send the resuls to csv/excel and put them on a presentation slide.
Feel free to copy/edit the code for any purpose bute take the code as is,
also no guarantees of any sort, but please report me of any possible errors.
"""
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np

def classification_report_df(y_true,y_pred,classes):
    clf_rep = precision_recall_fscore_support(y_true, y_pred)
    avgs=[]
    totalsum = np.sum(clf_rep[3])
    for i in range(0,3):
        avgs.append(np.sum(clf_rep[i]*clf_rep[3])/totalsum)

    avgs.append(totalsum)
    mylist = [list(x) for x in clf_rep]
    clf_rep_all = [x + [y] for x,y in zip(mylist,avgs)]
    indices = list(classes) +['avg/total']
    out_dict = {
                 "precision" :clf_rep_all[0]
                ,"recall" : clf_rep_all[1]
                ,"f1-score" : clf_rep_all[2]
                ,"support" : clf_rep_all[3]
                }
    out_df = pd.DataFrame(out_dict, index = indices)
    out_df[["precision","recall","f1-score"]]= out_df[["precision","recall","f1-score"]].apply(lambda x: round(x,2))
    return out_df
