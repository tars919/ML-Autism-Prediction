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

#creats a pie chart -> learn data is highly imblanced, hard time predicting positive class
#plt.pie(df['Class/ASD'].value_counts().values, autopct='%1.1f%%')
#plt.show()


#seperating the data based on the type
ints = []
objects = []
floats = []

for col in df.columns:
    if df[col].dtype == int:
        ints.append(col)
    elif df[col].dtype == object:
        objects.append(col)
    else:
        floats.append(col)


#id column will contain a unique value for each of the rows and class/asd can be removed since we already analyzed 
ints.remove('ID')
ints.remove('Class/ASD')


# Create a figure and an array of subplots with 4 rows and 3 columns, setting the overall size of the figure
fig, axes = plt.subplots(4, 3, figsize=(15, 15))

# Iterate through the columns in the 'ints' list using enumerate to get both the index and column name
for i, col in enumerate(ints):
    # Calculate the row and column indices for the current subplot
    row_idx, col_idx = divmod(i, 3)
    
    # Check if the calculated row index is within the bounds of the 'axes' array
    if row_idx < axes.shape[0]:
        # Create a countplot for the current column 'col' from the dataframe 'df'
        # The x-axis represents the categories in 'col', colored by the 'Class/ASD' column
        # Place the countplot in the corresponding subplot determined by 'row_idx' and 'col_idx'
        sb.countplot(data=df, x=col, hue='Class/ASD', ax=axes[row_idx, col_idx])

# Adjust the layout of the subplots to prevent overlap
plt.tight_layout()

# Display the plots
#plt.show()

#-----> Can conclude that if the score of some indicator is 0 the chance of that person not having autsims is quite high 
#except for the A10_score case

#do the same this

plt.subplots(figsize=(15, 30))

for i, col in enumerate(objects):
    plt.subplot(5, 3, i+1)
    sb.countplot(data=df, x=col, hue='Class/ASD', dodge=True)
    plt.xticks(rotation=60)

plt.tight_layout()
#plt.show()

#------> Can conclude age_desc is the same for all the points, used_app is a source of data leakage because it's  useless
# assume : males w autism is higher than females but not equal sample amount


# Create a figure with a specific size
plt.figure(figsize=(15, 5))

# Create a countplot using Seaborn to visualize the distribution of 'country_of_res' categories,
# with different colors representing the 'Class/ASD' categories
sb.countplot(data=df, x='contry_of_res', hue='Class/ASD')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Display the plot
plt.show()

#---> Some places have 50% of yhe data have autism -> so geography plays a role in giving an idea 


plt.subplots(figsize = (15,5))

for i, col in enumerate(floats):
    plt.subplot(1,2,i+1)
    sb.distplot(df[col])
plt.tight_layout()
#plt.show()

#------> Both the charts are skewed left is positively and right is negatively

plt.subplots(figsize = (15,5))

for i, col in enumerate(floats):
    plt.subplot(1,2,i+1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()

#--> This showed us the outliers in the result column -> there is none so the shape will be the same value of the same total datapoints

df = df[df['result']>-5]
#print(df.shape)

#- - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - -- - -- - - - - 
#FEATURE ENGINEERING








