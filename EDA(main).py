# 1. Problem Statement : find the Problem and Analyse it.
# finding the person having the diabaties

#2.load and import the all Libraries required.
#load all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#3.Load the dataset
#load the data
data = pd.read_csv('diabetes1.csv')
data

#4.Domain Analysis.(Basic Checks)
data.shape
data.head()
data.tail()
data.columns
data.info()  # Data Information.
data.describe()  # Statistical Analysis.

#5.EDA
#A. Univariate analsyis - analyse the single column
# 1.cat / descrete - countplot 
# 2.continuous -> histogram / box 
# target column
# count plot
#plt.figure(figsize=(5,3))
sns.countplot(x=data['Outcome'])
sns.histplot()
#SUBPLOT
plt.figure(figsize=(10,10),facecolor='pink')
plot = 1
for i in data:
    if plot < 10:
        ax = plt.subplot(3,3,plot)
        sns.distplot(data[i])
        plt.xlabel(i)
    plot+=1
plt.tight_layout()
plt.show()
sns.histplot(x=data['Glucose'],hue=data['Outcome'])

#B.Bivarite - 2 col analysis
#1.num vs num  -> scatter, line, rel
#2.num vs cat   -> count(hue), bar, box (hue)
sns.barplot(y=data['Glucose'],hue=data['Outcome'])
sns.histplot(x=data['Glucose'],hue=data['Outcome'])
sns.boxplot(x=data['Glucose'],hue=data['Outcome'])

#C.Multivariate
#- Pai#sns.pairplot(data)
sns.pairplot(data , hue='Outcome')rplot

# Data Preprocessing
#null values and replacing them.
data.isnull().sum()
data.duplicated().sum()
data['SkinThickness'].replace(0, data['SkinThickness'].median(),inplace = True)
data['Insulin'].replace(0, data['Insulin'].median(),inplace = True)
data['BMI'].replace(0, data['BMI'].median(),inplace = True)

#Feature Selection.
#drop the irrlevant columns
# co relation of data
sns.heatmap(data.drop('Outcome',axis=1).corr(),annot=True)

#Model Building.
x = data.drop('Outcome',axis=1)
y = data[['Outcome']]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x,y,test_size=0.30,random_state=2)

#Data Tranformation --(Scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = pd.DataFrame(sc.fit_transform(x_train),columns=x_train.columns)
x_test = pd.DataFrame(sc.transform(x_test),columns=x_test.columns)
x_train
x_test
#Model Creation
#apply the logistic Regression
#import the algo
from sklearn.linear_model import LogisticRegression
# init
model = LogisticRegression()
# train the model
model.fit(x_train, y_train)
# test the model with new records
# x_test
y_pred = model.predict(x_test)
y_pred
#Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score,classification_report
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
