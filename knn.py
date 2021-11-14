# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 22:37:04 2021

@author: issam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

df=pd.read_csv("heart2.csv", index_col=0)

print("Head : \n")
print(df.head())
print("Describe : \n")
print(df.describe())
print("Info : \n")
print(df.info())

fig=plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot= True, cmap='Blues')

#Training dataset : 
df_train=df.iloc[:225,:]
#Testing dataset :
df_test=df.iloc[225:,:]

k=5

def distance(p1, p2) : #Compute the distance between 2 patients p1, p2 : np.arrays
    d=0
    for i in range(len(p1)):
       d+=(p1[i]-p2[i])**2
    return math.sqrt(d)

def k_smallest(l): #Returns the k index of the k smallest values in an np.array l
    return np.argsort(l)[:k]

# Function that return the class that repeats the most within the k smallest value of the sorted listed above.

# Function that predicts the class based on the function above.

""" 
df.iloc[0,:-1] : return the first row without the target value
df.iloc[0,:-1].to_numpy() : convert as pd.series to an np.array
"""