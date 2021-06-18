# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:59:11 2021

@author: DELL
"""
Q) classify whether the claim is with Attorney or not.

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
claimants=pd.read_csv("claimants.csv")
claimants.head()
claimants.columns
#'CASENUM', 'ATTORNEY', 'CLMSEX', 'CLMINSUR', 'SEATBELT', 'CLMAGE','LOSS'

#Removing the CASENUM as it does not play any important role in the
#classifying the whether the claim is with Attorney or not
claimants.drop(["CASENUM"],axis=1,inplace=True)
claimants.head()
claimants.columns
#ATTORNEY', 'CLMSEX', 'CLMINSUR', 'SEATBELT', 'CLMAGE', 'LOSS

#Visualization
#plot b/w CLMSEX and CLMAGE
import seaborn as sns
sns.boxplot(x="CLMSEX",y="CLMAGE",data=claimants)
plt.boxplot(claimants.LOSS)
plt.hist(claimants.LOSS)
claimants.describe()

#To check the missing values
claimants.isna().sum()
#ATTORNEY=0,CLMSEX=12,CLMINSUR=41,SEATBELT=48,CLMAGE=189,LOSS=0

#IMPUTATION
#checking the most occuring values in the columns
claimants.ATTORNEY.value_counts()
claimants.CLMSEX.value_counts()
claimants.SEATBELT.value_counts()

#Imputing the missing values with most occuring values
claimants.SEATBELT.value_counts()
#As the most occuring number is 0
claimants.SEATBELT=claimants.SEATBELT.fillna(claimants.SEATBELT.value_counts().index[0])

claimants.CLMSEX.value_counts()
#As the most occuring number is 1
claimants.CLMSEX=claimants.CLMSEX.fillna(claimants.CLMSEX.value_counts().index[1])
claimants.isna().sum()

claimants.CLMINSUR.value_counts()
#most occuring value is 1
claimants.CLMINSUR=claimants.CLMINSUR.fillna(claimants.CLMINSUR.value_counts().index[1])
claimants.isna().sum()

#As AGE is having many numbers so we will impute the missing values with the mean of the AGE column
claimants.CLMAGE=claimants.CLMAGE.fillna(claimants.CLMAGE.mean())

#Final check to see the missing values
claimants.isna().sum()
#There are no missing values

#Build the model
from scipy import stats
import statsmodels.formula.api as sm
model=sm.logit("ATTORNEY~CLMSEX+CLMINSUR+SEATBELT+CLMAGE+LOSS",data=claimants).fit()
model.summary()

pred=model.predict(claimants)

claimants["y_pred"]=pred

#creating a new column Att_values and filling with 0
claimants["Att_values"]=0
#taking a threshold value as 0.5 and above will be treated as correct value(1)
claimants.loc[pred>=0.5,"Att_values"]=1
claimants.Att_values

#checking the accuracy
from sklearn.metrics import classification_report
classification_report(claimants.ATTORNEY,claimants.Att_values)
#or confusion_matrix
confusion_matrix=pd.crosstab(claimants["ATTORNEY"],claimants.Att_values)
confusion_matrix
accuracy=(505+442)/(505+442+243+150)
accuracy
#0.71

#ROC curve
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(claimants.ATTORNEY,pred)
roc_auc=metrics.auc(fpr,tpr)
#0.76

#plot of roc_auc
plt.plot(fpr,tpr);plt.xlabel("fpr");plt.ylabel("tpr")

#We see that most of the data is under the curve














