#!/usr/bin/env python
# coding: utf-8
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

numerical_columns = user.select_dtypes(include='number')
for col in numerical_columns:
    user[col].fillna(user[col].mean(), inplace=True)
numerical_columns.isnull().sum()
numerical_columns = numerical_columns.drop(columns = ["FIELDEMAILLAST","FIELDIMPRESSIONVALUECOUNT","EVENTPUZZLE_SHUFFLECOUNT", "FIELDACTIONNAMELAST","FIELDUSERLEVELCOUNT", "FIELDUSERLEVELFIRST", "FIELDUSERLEVELLAST"],axis = 1)

def find_best_fitting_features(data, y_variable, k):
    selected_features = []
    remaining_features = set(data.columns)
    labels = set(['D7_RETENTION', "TIME_DIFF"])
    remaining_features -= labels
    best_accuracy = 0
    for i in range(k):
        best_feature = None
        
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X = data[current_features]
            y = y_variable.values
            
            model = LogisticRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature
        print(f"Round {i} best acc: {best_accuracy}")
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
    print(f"Best acc: {best_accuracy}")
    return tuple(selected_features)


data = numerical_columns
y_variable = user['D7_RETENTION']

selected_features = find_best_fitting_features(data, y_variable, 10)
print("Best selected_feature:",selected_features)

X = numerical_columns[list(selected_features)]
y = user['D7_RETENTION']

model = LogisticRegression()
model.fit(X, y)
coefficients = model.coef_[0]

#feature importance
feature_importance = abs(coefficients)
feature_importance /= feature_importance.sum()
feature_importance_dict = dict(zip(selected_features, feature_importance))
for feature, importance in feature_importance_dict.items():
    print(f"{feature}: {importance}")

