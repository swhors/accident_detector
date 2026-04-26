from sklearn.svm import SVC


def train_svm(X_train_scaled, y_train):
    svm_model = SVC(kernel='rbf', probability=True) # 확률 예측을 위해 True 설정
    svm_model.fit(X_train_scaled, y_train) # 스케일링된 데이터 사용 필수
    # svm_pred = svm_model.predict(X_test_scaled)
    return svm_model

