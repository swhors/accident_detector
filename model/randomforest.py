from sklearn.ensemble import RandomForestClassifier


def train_randomforest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    # rf_pred = rf_model.predict(X_test)
    return rf_model

