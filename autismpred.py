#INFO
# - Go to kaggle import the dataset and save it as train.csv
# - To find the dataset search A1_score and it should come up -> autism spectrum prediction challenge


#Needed imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

#commenting out the print statements when needed
#loading the dataset into the pandas data frame 
df = pd.read_csv('train.csv')

#then printing its first 5 rows
#print(df.head())

#checking the size of the dataset
#print(df.shape)

#to check which column of the dataset contains which type of data
#print(df.info())

#Shows us in each column that there are no null values
#print(df.describe().T)



#- - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - -- - -- - - - - 
#DATA CLEANING: This step will remove ouliers, null value imputation, and removing discrepancies

#displays all the ethnicities and how many of each ethnicity
#print(df['ethnicity'].value_counts())

#this is counting the different relations and how many of each -> shows us theres a ? and others when there should just be one of that catagory
#print(df['relation'].value_counts())
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Cleaning the dats ~~~ and converting yes to 0 and no to 1
#groups ? and others into the same Others category
df = df.replace({'yes':1, 'no':0, '?':'Others', 'others':'Others'})

#- - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - -- - -- - - - - 
#EXPLORATORY DATA ANALYSIS







