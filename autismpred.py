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
#FEATURE ENGINEERING: helps derive valuable features from existsing ones/ give deeper insights to the data

#this function make groups by taking age as the parameter
def convertAge(age):
    if age <4:
        return 'Todler'
    elif age <12:
        return 'Kid'
    elif age <18:
        return 'Teenager'
    elif age <40:
        return 'Young'
    else:
        return "Senior"
    

df['ageGroup'] = df['age'].apply(convertAge)

sb.countplot(x=df['ageGroup'], hue=df['Class/ASD'])
plt.show()

#------> can  conclude that young and todler groups will have a lower chance of having autism need to divide blue by orange

#summing up the clinical scores given from A1 to A10
def add_feature(data):

    #creating a column with all values zero
    data['sum_score'] = 0
    for col in data.loc[:,'A1_Score':'A10_Score'].columns:

        #updating the 'sum_score' value with scores from A1 to A10
        data['sum_score'] += data[col]

    #creating a random data using the below three columns
    data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']

    return data

df = add_feature(df)

sb.countplot(x=df['sum_score'], hue=df['Class/ASD'])
plt.show()

#-----------> higher the sum score highter the chances of having autism is higher as well and similarly for lower sum scores that are for less that 5 it is rare they have autism



#applying log transformations to remove the skewness of the data

df['age'] = df['age'].apply(lambda x: np.log(x))

sb.distplot(df['age'])
plt.show()

#N-----> now there is no skew in the data

def encode_labels(data):
    for col in data.columns:

        #here we check the datatype 
        #if object then we encode it 

        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    return data

#self calling
df = encode_labels(df)

#making a heatmap to visualize the correlation matrix
plt.figure(figsize=(10,10))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

#-----> can conclude that there is only one highlt correlated features which we will remove before training the model on this data
#- - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - -- - -- - - - - 

#MODEL TRAINING: here we will separate the features and target variable and split them into training and the testing data by using which performs sbest ofn validation data

removal = ['ID','age_desc', 'used_app_before','austim']
features = df.drop(removal + ['Class/ASD'], axis=1)
target = df['Class/ASD']

#splitting the data
#normalizing the data to obtain stable and fast training 


X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size = 0.2, random_state=10)

#as the data was highly imbalanced we will balance it by adding repetitive rows of minority class
ros = RandomOverSampler(sampling_strategy='minority',random_state=0)
X, Y = ros.fit_resample(X_train, Y_train)
#----> (992,20) (992,)
#print(X.shape, Y.shape)


#normalizing the features for stable and fast training
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

#now training some state of the art mlmodels and compare to see which fits better with our data
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for model in models:
    model.fit(X,Y)

    print(f'{model} : ')
    print('Training Accuracy: ', metrics.roc_auc_score(Y, model.predict(X)))
    print('Validation Accuracy: ', metrics.roc_auc_score(Y_val, model.predict(X_val)))
    print()

#--------> from these scores we can see that logistic regression and SVC have lower numbers so they perform better bc there is less difference between the validation and training data

# Assuming models[0] is your trained classifier model and X_val, Y_val are your validation data
# Generate predictions using the trained model
predictions = models[0].predict(X_val)

# Compute the confusion matrix using the true labels (Y_val) and predictions
cm = confusion_matrix(Y_val, predictions)

# Print or visualize the confusion matrix as needed
print("Confusion Matrix:")
print(cm)

# Optionally, you can plot the confusion matrix
plt.figure(figsize=(8, 6))
sb.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


#######------> NOW to calculate the actual accuracy percentage of the model 
#____________________________________________________________________

# Calculate accuracy from the confusion matrix
accuracy = accuracy_score(Y_val, predictions)

#print out the actual accuracy percent and converts from decimal to percent
print("Accuracy:", accuracy * 100, '%')






#---------
#CONCLUDE
#---------
#can conclude that if the score is blue then the chance of the not having autism is high except for some outlier cases
#ML shows that geography also will give an idea 
#the variety of graphs will allow us to learn different parts of information using a csv file 
#this produces 84% accurate results






