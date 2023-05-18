#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import matplotlib
matplotlib.use('Agg')


# In[3]:

datapath = "heart_cleveland_upload.csv"
data = pd.read_csv(datapath)


# In[4]:


data


# In[ ]:





# In[5]:


data.info()


# Our aim here is to analyze our data and report our findings through visualizations and the code below allows us to check if we have any missing values in our dataset before going further with the analysis.

# In[6]:


data.isnull().sum()


# #### creating a copy of dataset so that will not affect our original dataset.
# 

# In[7]:


heart_df = data.copy()


# In[8]:


heart_df.describe().T ## We Use T to round off the values


# In[9]:


#Rename Our Condition column to Target column
heart_df = heart_df.rename(columns={'condition':'target'})
print(heart_df.head(1))


# # Exploratory data analysis (EDA) 

# In[10]:


#We Will be checking the relationship among the variable
corr = heart_df.corr()
plt.figure(figsize = (15,15))
sns.heatmap(corr, annot = True , fmt = ".1f", linewidths = .7, cmap="Blues")


# In[11]:


#fixing our data in x and y. Here y contains target data and X contains rest all the features.
x= heart_df.drop(columns= 'target')
y= heart_df.target


# In[12]:


sns.countplot(y)


# In[13]:


heart_df.info()


# In[14]:


cat_cols= ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal','target']

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak','target']



# In[15]:


print(cat_cols)
print(num_cols)


# In[ ]:





# In[16]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['sex'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='sex')


# In[17]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['cp'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='Chest Pain')


# In[18]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['fbs'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='Fasting Blood Sugar')


# In[19]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['restecg'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='Rest ECG')


# In[20]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['exang'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='Exang')


# In[21]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['slope'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='slope')


# In[22]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['ca'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='CA')


# In[23]:


fig, ax = plt.subplots()
sns.countplot(ax = ax, data = heart_df, x = heart_df['thal'], hue = "target")
for label in ax.containers:
    ax.bar_label(label)
ax.set(ylabel='Counts', title='THAL ')


# ## Numeric Feature Analysis

# In[24]:


def histPlot(num):
    sns.histplot(data = heart_df, x = num, bins = 50, kde = True)
    print("{} distribution with hist:".format(num))
    plt.show()


# In[25]:


num_cols


# In[26]:


sns.histplot(data = heart_df, x = heart_df['age'], bins = 50, kde = True)
#print("{} distribution with hist:".format(heart_df['age']))
plt.show()


# In[27]:


sns.histplot(data = heart_df, x = heart_df['chol'], bins = 50, kde = True)
#print("{} distribution with hist:".format(heart_df['age']))
plt.show()


# In[28]:


sns.histplot(data = heart_df, x = heart_df['trestbps'], bins = 50, kde = True)
#print("{} distribution with hist:".format(heart_df['age']))
plt.show()


# In[29]:


sns.histplot(data = heart_df, x = heart_df['thalach'], bins = 50, kde = True)
#print("{} distribution with hist:".format(heart_df['age']))
plt.show()


# In[30]:


sns.histplot(data = heart_df, x = heart_df['oldpeak'], bins = 50, kde = True)
#print("{} distribution with hist:".format(heart_df['age']))
plt.show()


# In[31]:


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

# In[32]:


cat_cols= ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']



# In[33]:


#heart_df = pd.get_dummies(heart_df,columns=cat_cols,drop_first=True)


# In[34]:


heart_df.info()


# In[35]:


x = heart_df.drop(['target'],axis=1)
y = heart_df[['target']]


# In[ ]:





# ## Spliting our data set into train and test set

# In[36]:


# Spliting our data set into train and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)


# In[37]:


x_train.info()


# In[ ]:





# In[38]:


scaler = StandardScaler()
scaler


# In[39]:


x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.fit_transform(x_test)


# ## Modeling

# ### Logistic Regression

# In[40]:


model_lr=LogisticRegression()


# In[41]:


y_train = np.array(y_train)
y_traintmp = y_train.ravel()
y_train = np.array(y_traintmp).astype(int)
type(y_train)


# In[42]:


model_lr.fit(x_train_scaler,y_train)


# In[43]:


y_test_pred = model_lr.predict(x_test_scaler)


# In[44]:


p=model_lr.score(x_test_scaler,y_test_pred)


# In[45]:


#filename = 'Heart_Attack_Prediction_LR.h5'
pickle.dump(model_lr, open('model_lr.pkl', 'wb'))


# In[46]:


type(y_test), type(y_test_pred)


# In[47]:


from sklearn.metrics import accuracy_score
accuracy_score=accuracy_score(y_test,y_test_pred)
print("Test score: {}".format(accuracy_score))


# In[48]:


print(confusion_matrix(y_test, y_test_pred))


# In[49]:


y_pred_prob = model_lr.predict_proba(x_test_scaler)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])


# In[50]:


plt.plot([0,1],[0,1],"k--")
plt.plot(fpr, tpr, label = "Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.show()


# ### Creating K-Nearest-Neighbor classifier

# In[51]:


sel=SelectFromModel(RandomForestClassifier())


# In[52]:


sel.fit(x_train_scaler,y_train)


# In[53]:


sel.get_support()


# In[54]:


selected_feat= x_train.columns[(sel.get_support())]
selected_feat


# In[55]:


#y_train = np.array(y_train)


# In[56]:


#type(x_train_scaler), type(y_train)


# In[57]:


#x_train_scaler = pd.DataFrame(x_train_scaler)
#type(x_train_scaler)


# In[58]:


model_RMC=RandomForestClassifier()
model_RMC.fit(x_train_scaler, y_train)
y_test_pred = model_RMC.predict(x_test_scaler)


from sklearn.metrics import accuracy_score
accuracy_score=accuracy_score(y_test,y_test_pred)
print("Test score: {}".format(accuracy_score))


# In[59]:


#filename = 'Heart_Attack_Prediction_LR.h5'
pickle.dump(model_RMC, open('model_RMC.pkl', 'wb'))


# ## XGBClassifier

# In[60]:


model_xgb= XGBClassifier()


# In[61]:


model_xgb.fit(x_train_scaler,y_train)


# In[62]:


y_test_pred = model_xgb.predict(x_test_scaler)


# In[63]:


from sklearn.metrics import accuracy_score
accuracy_score=accuracy_score(y_test,y_test_pred)
print("Test score: {}".format(accuracy_score))


# In[64]:


#filename = 'Heart_Attack_Prediction_LR.h5'
pickle.dump(model_xgb, open('model_xgb.pkl', 'wb'))


# In[ ]:




