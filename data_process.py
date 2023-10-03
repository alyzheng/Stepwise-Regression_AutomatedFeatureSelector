#!/usr/bin/env python
# coding: utf-8

# In[549]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[550]:


user = pd.read_csv("user_metrics.csv")


# In[553]:


user['USER_LAST_SEEN_DATE'] = pd.to_datetime(user['USER_LAST_SEEN_DATE'])
user['USER_FIRST_SEEN_DATE'] = pd.to_datetime(user['USER_FIRST_SEEN_DATE'])


# In[554]:


user['TIME_DIFF'] = (user['USER_LAST_SEEN_DATE'] - user['USER_FIRST_SEEN_DATE']).dt.days


# In[555]:


user['D7_RETENTION'] = user['TIME_DIFF'].apply(lambda x: 1 if x >= 7 else 0)


# In[557]:


sns.countplot(data=user, x='D7_RETENTION')
plt.title('Distribution of D7_RETENTION')
plt.show()


# In[558]:


sns.histplot(data=user, x='EVENTAPP_LAUNCHCOUNT')
plt.show()


# In[559]:


sns.boxplot(data=user, y='EVENTCLIENTDEVICECOUNT', x='D7_RETENTION')
plt.show()


# In[560]:


sns.scatterplot(data=user, x='EVENTGAMESTARTEDCOUNT', y='EVENTADSHOWCOUNT', hue='D7_RETENTION')
plt.show()


# In[561]:


correlation_matrix = user[['EVENTAPP_LAUNCHCOUNT', 'EVENTAPP_EXITCOUNT', 'EVENTAPP_FINISH_LOADINGCOUNT',
                        'EVENTCLIENTDEVICECOUNT', 'EVENTGAMESTARTEDCOUNT', 'EVENTADSHOWCOUNT',
                        'EVENTNEWPLAYERCOUNT', 'EVENTTUTORIAL_STEP_COMPLETECOUNT',
                        'EVENTUI_POPUPCOUNT', 'TOTALDAYSPLAYED', 'D7_RETENTION']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[562]:


features = ["EVENTAPP_LAUNCHCOUNT","EVENTAPP_EXITCOUNT","EVENTAPP_FINISH_LOADINGCOUNT","EVENTCLIENTDEVICECOUNT","EVENTGAMESTARTEDCOUNT","EVENTADSHOWCOUNT","EVENTNEWPLAYERCOUNT","EVENTTUTORIAL_STEP_COMPLETECOUNT","EVENTUI_POPUPCOUNT","TOTALDAYSPLAYED" ]


# In[563]:


user[features].isnull().sum()


# In[565]:


user["EVENTAPP_LAUNCHCOUNT"].fillna(user["EVENTAPP_LAUNCHCOUNT"].mean(), inplace = True)
user["EVENTAPP_EXITCOUNT"].fillna(user["EVENTAPP_EXITCOUNT"].mean(), inplace = True)
user["EVENTAPP_FINISH_LOADINGCOUNT"].fillna(user["EVENTAPP_FINISH_LOADINGCOUNT"].mean(), inplace = True)
user["EVENTCLIENTDEVICECOUNT"].fillna(user["EVENTCLIENTDEVICECOUNT"].mean(), inplace = True)
user["EVENTGAMESTARTEDCOUNT"].fillna(user["EVENTGAMESTARTEDCOUNT"].mean(), inplace = True)
user["EVENTNEWPLAYERCOUNT"].fillna(user["EVENTNEWPLAYERCOUNT"].mean(), inplace = True)
user["EVENTTUTORIAL_STEP_COMPLETECOUNT"].fillna(user["EVENTTUTORIAL_STEP_COMPLETECOUNT"].mean(), inplace = True)
user["EVENTUI_POPUPCOUNT"].fillna(user["EVENTUI_POPUPCOUNT"].mean(), inplace = True)
user["TOTALDAYSPLAYED"].fillna(user["TOTALDAYSPLAYED"].mean(), inplace = True)
user["EVENTADSHOWCOUNT"].fillna(user["EVENTADSHOWCOUNT"].mean(), inplace = True)


# In[568]:


target = 'D7_RETENTION'


# In[569]:


user[features]


# In[570]:


scaler = StandardScaler()


# In[571]:


scaler.fit(user[features])


# In[572]:


user[features] = scaler.transform(user[features])

