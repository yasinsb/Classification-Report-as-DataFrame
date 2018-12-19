# example for classification_report_df function

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from classification_report_df import classification_report_df

from random import randint, uniform, normalvariate

#----generating some random data to use for classification---#
y = np.array([randint(0,1) for i in range(100)])
y = y.reshape(-1,1)

X = np.array([y[i]*2+normalvariate(0,1) for i in range(100)])
X = X.reshape(-1,1)

#Using Logistic Regression to predict. Note than any classifier could be used
LR_CLF = LogisticRegression()
LR_CLF.fit(X,y.ravel())
y_pred = LR_CLF.predict(X)

#the desired output in a dataframe
CR_df = classification_report_df(y,y_pred,['A','B'])

#to get the recall for A class you can use:
#CR_df.loc['A']['recall']