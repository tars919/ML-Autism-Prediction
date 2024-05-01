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

#loading the dataset into the pandas data frame 
#then printing its first 5 rows
df = pd.read_csv('train.csv')
print(df.head())


