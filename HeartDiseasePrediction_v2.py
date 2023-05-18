#!/usr/bin/env python
# coding: utf-8

# In[4]:


#pip install xgboost
import pandas as pd
import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier 


# In[5]:


import matplotlib
matplotlib.use('Agg')


# In[6]:


data = pd.read_csv('/Users/aneerajbidlan/Downloads/Problem Statement 1/heart_cleveland_upload.csv')


# In[7]:


data


# In[8]:


data.info()


# Our aim here is to analyze our data and report our findings through visualizations and the code below allows us to check if we have any missing values in our dataset before going further with the analysis.

# In[9]:


data.isnull().sum()


# #### creating a copy of dataset so that will not affect our original dataset.
# 

# In[10]:


heart_df = data.copy()


# In[11]:


heart_df.describe().T ## We Use T to round off the values


# In[12]:


#Rename Our Condition column to Target column
heart_df = heart_df.rename(columns={'condition':'target'})
print(heart_df.head(1))


# # Exploratory data analysis (EDA) 

# In[13]:


#We Will be checking the relationship among the variable
corr = heart_df.corr()
plt.figure(figsize = (15,15))
sns.heatmap(corr, annot = True , fmt = ".1f", linewidths = .7, cmap="Blues")


# In[14]:


#fixing our data in x and y. Here y contains target data and X contains rest all the features.
x= heart_df.drop(columns= 'target')
y= heart_df.target


# In[15]:


sns.countplot(y)


# In[16]:


heart_df.info()


# In[17]:


cat_cols= ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal','target']

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak','target']



# In[18]:


print(cat_cols)
print(num_cols)


# In[ ]:





# In[19]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['sex'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='sex')


# In[20]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['cp'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='Chest Pain')


# In[21]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['fbs'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='Fasting Blood Sugar')


# In[22]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['restecg'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='Rest ECG')


# In[23]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['exang'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='Exang')


# In[24]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['slope'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='slope')


# In[25]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['ca'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='CA')


# In[26]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['thal'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='THAL ')


# ## Numeric Feature Analysis

# In[27]:


def histPlot(num):
    sns.histplot(data = heart_df, x = num, bins = 50, kde = True)
    print("{} distribution with hist:".format(num))
    plt.show()


# In[28]:


num_cols


# In[29]:


sns.histplot(data = heart_df, x = heart_df['age'], bins = 50, kde = True)
#print("{} distribution with hist:".format(heart_df['age']))
plt.show()


# In[30]:


sns.histplot(data = heart_df, x = heart_df['chol'], bins = 50, kde = True)
#print("{} distribution with hist:".format(heart_df['age']))
plt.show()


# In[31]:


sns.histplot(data = heart_df, x = heart_df['trestbps'], bins = 50, kde = True)
#print("{} distribution with hist:".format(heart_df['age']))
plt.show()


# In[32]:


sns.histplot(data = heart_df, x = heart_df['thalach'], bins = 50, kde = True)
#print("{} distribution with hist:".format(heart_df['age']))
plt.show()


# In[33]:


sns.histplot(data = heart_df, x = heart_df['oldpeak'], bins = 50, kde = True)
#print("{} distribution with hist:".format(heart_df['age']))
plt.show()


# In[34]:


for i in ['trestbps', 'chol', 'thalach']:
    plt.figure(figsize=(12,5))
    sns.lineplot(y=i,x="age",data=heart_df)
    plt.title(f"{i} WITH AGE",fontsize=20)
    plt.xlabel(i,fontsize=15)
    plt.ylabel(i,fontsize=15)
    plt.show()


# There is no strong Relationship with age and heart attack.So we can't say with Increasing the Age There is high Chance of Heart attack or Low Chance of Heart Attack.
# 
# There is high chance of Increase in Blood Pressure in the body With Increase in Age.
# 
# There is high chance of Increase in Cholestrol Level in the body with increase in Age.
# 
# There is high chance of Increase in Heart Rate in the body with increase in Age

# ## Encoding the data

# In[46]:


cat_cols= ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']



# In[47]:


heart_df = pd.get_dummies(heart_df,columns=cat_cols,drop_first=True)


# In[48]:


heart_df.info()


# In[49]:


x = heart_df.drop(['target'],axis=1)
y = heart_df[['target']]


# ## Spliting our data set into train and test set

# In[50]:


# Spliting our data set into train and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)


# In[51]:


x_train.info()


# In[52]:


scaler = StandardScaler()
scaler


# In[53]:


x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.fit_transform(x_test)


# ## Modeling

# ### Logistic Regression

# In[54]:


model_lr=LogisticRegression()


# In[55]:


y_train = np.array(y_train)
y_traintmp = y_train.ravel()
y_train = np.array(y_traintmp).astype(int)
type(y_train)


# In[56]:


model_lr.fit(x_train_scaler,y_train)


# In[57]:


y_test_pred = model_lr.predict(x_test_scaler)


# In[58]:


p=model_lr.score(x_test_scaler,y_test_pred)


# In[59]:


filename = 'Heart_Attack_Prediction_LR.h5'
pickle.dump(model_lr, open(filename, 'wb'))


# In[60]:


type(y_test), type(y_test_pred)


# In[61]:


from sklearn.metrics import accuracy_score
accuracy_score=accuracy_score(y_test,y_test_pred)
print("Test score: {}".format(accuracy_score))


# In[62]:


print(confusion_matrix(y_test, y_test_pred))


# In[63]:


y_pred_prob = model_lr.predict_proba(x_test_scaler)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])


# In[64]:


plt.plot([0,1],[0,1],"k--")
plt.plot(fpr, tpr, label = "Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.show()


# ### Creating K-Nearest-Neighbor classifier

# In[65]:


sel=SelectFromModel(RandomForestClassifier())


# In[66]:


sel.fit(x_train_scaler,y_train)


# In[67]:


sel.get_support()


# In[68]:


selected_feat= x_train.columns[(sel.get_support())]
selected_feat


# In[69]:


#y_train = np.array(y_train)


# In[70]:


#type(x_train_scaler), type(y_train)


# In[71]:


#x_train_scaler = pd.DataFrame(x_train_scaler)
#type(x_train_scaler)


# In[72]:


model_RMC=RandomForestClassifier()
model_RMC.fit(x_train_scaler, y_train)
y_test_pred = model_RMC.predict(x_test_scaler)
type(y_test), type(y_test_pred)

from sklearn.metrics import accuracy_score
accuracy_score=accuracy_score(y_test,y_test_pred)
print("Test score: {}".format(accuracy_score))


# ## XGBClassifier

# In[73]:


model_xgb= XGBClassifier()


# In[74]:


model_xgb.fit(x_train_scaler,y_train)


# In[75]:


y_test_pred = model_xgb.predict(x_test_scaler)


# In[76]:


from sklearn.metrics import accuracy_score
accuracy_score=accuracy_score(y_test,y_test_pred)
print("Test score: {}".format(accuracy_score))


# In[ ]:





# In[ ]:




