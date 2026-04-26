from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb_model.fit(X_train, y_train)
    return xgb_model

