#!/usr/bin/env python
# coding: utf-8
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

user = pd.read_csv("user_metrics.csv")
user['USER_LAST_SEEN_DATE'] = pd.to_datetime(user['USER_LAST_SEEN_DATE'])
user['USER_FIRST_SEEN_DATE'] = pd.to_datetime(user['USER_FIRST_SEEN_DATE'])
user['TIME_DIFF'] = (user['USER_LAST_SEEN_DATE'] - user['USER_FIRST_SEEN_DATE']).dt.days
user['D7_RETENTION'] = user['TIME_DIFF'].apply(lambda x: 1 if x >= 7 else 0)

#exploratory data analysis
sns.countplot(data=user, x='D7_RETENTION')
plt.title('Distribution of D7_RETENTION')
plt.show()

sns.histplot(data=user, x='EVENTAPP_LAUNCHCOUNT')
plt.show()

sns.boxplot(data=user, y='EVENTCLIENTDEVICECOUNT', x='D7_RETENTION')
plt.show()

sns.scatterplot(data=user, x='EVENTGAMESTARTEDCOUNT', y='EVENTADSHOWCOUNT', hue='D7_RETENTION')
plt.show()

correlation_matrix = user[['EVENTAPP_LAUNCHCOUNT', 'EVENTAPP_EXITCOUNT', 'EVENTAPP_FINISH_LOADINGCOUNT',
                        'EVENTCLIENTDEVICECOUNT', 'EVENTGAMESTARTEDCOUNT', 'EVENTADSHOWCOUNT',
                        'EVENTNEWPLAYERCOUNT', 'EVENTTUTORIAL_STEP_COMPLETECOUNT',
                        'EVENTUI_POPUPCOUNT', 'TOTALDAYSPLAYED', 'D7_RETENTION']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

#feature engineering
features = ["EVENTAPP_LAUNCHCOUNT","EVENTAPP_EXITCOUNT","EVENTAPP_FINISH_LOADINGCOUNT","EVENTCLIENTDEVICECOUNT","EVENTGAMESTARTEDCOUNT","EVENTADSHOWCOUNT","EVENTNEWPLAYERCOUNT","EVENTTUTORIAL_STEP_COMPLETECOUNT","EVENTUI_POPUPCOUNT","TOTALDAYSPLAYED" ]
user[features].isnull().sum()

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

#set up y label
target = 'D7_RETENTION'
user[features]
scaler = StandardScaler()
scaler.fit(user[features])
user[features] = scaler.transform(user[features])

