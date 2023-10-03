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
