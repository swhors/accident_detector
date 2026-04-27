"""
랜덤 포레스트 모델 학습 및 반환
"""
from sklearn.ensemble import RandomForestClassifier


def train_randomforest(X_train, y_train):
    """
    랜덤 포레스트 모델 학습 및 반환
    
    :param X_train: 학습용 입력 데이터 (스케일링 전)
    :param y_train: 학습용 타겟 레이블
    :return: 학습된 랜덤 포레스트 모델
    """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    # rf_pred = rf_model.predict(X_test)
    return rf_model

