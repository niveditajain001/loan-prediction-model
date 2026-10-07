import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
df=pd.read_csv('dataset/train_u6lujuX_CVtuZ9i.csv')
print (df)
print(df.info())
print(df.describe())
pd.set_option('display.float_format','{:.2f}'.format)
print(df.describe())
print(df.head())
print(df['Loan_ID'].count())
print(df['Loan_Status'].value_counts())#if spaces error occurs
print(df.columns)
df.columns=df.columns.str.strip() #to remove white space from column names 
print(df['Education'].unique())#prints unique values of education column
df['Education']=df['Education'].str.strip()
df['Self_Employed']=df['Self_Employed'].str.strip()
df['Gender']=df['Gender'].str.strip()
df['Married']=df['Married'].str.strip()
df['Loan_Status'] = df['Loan_Status'].str.strip()
print(df['Loan_Status'].value_counts(normalize=True)*100)#in percentage
print(df.isnull().sum())
df.dropna(inplace=True)#remove null containing rows
#Loan_ID               0 
# Gender               13
# Married               3
# Dependents           15
# Education             0
# Self_Employed        32
# ApplicantIncome       0
# CoapplicantIncome     0
# LoanAmount           22
# Loan_Amount_Term     14
# Credit_History       50
# Property_Area         0
# Loan_Status           0
# dtype: int64
# (480, 13)
print(df.shape)
#now change all categorical to numerical data types
sns.countplot(x='Loan_Status',data=df)
# plt.show()
df.replace({'Loan_Status':{'Y':1,'N':0},'Married': {'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'Yes':1,'No':0},'Education':{'Graduate':1,'Not Graduate':0},'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2}},inplace=True)
df=df.replace(to_replace='3+',value=4)
df['Dependents'] = pd.to_numeric(df['Dependents'])
# df.drop('Loan_ID')
# df.drop('Loan_Status')
#separating data and label
X=df.drop(columns=['Loan_ID','Loan_Status'])#for colums axis=1 rows=0)
Y=df['Loan_Status'].astype('int')#label for dataset
print(X)
print(Y)
#for train test split data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)#stratify to split lables equally oterwise one might have ore 0 othr less random.. to split data in same way
print(X.shape,X_train.shape,X_test.shape)