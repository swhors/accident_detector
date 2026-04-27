"""
XGBoost 모델 학습 및 반환
"""
from xgboost import XGBClassifier


def train_xgboost(X_train, y_train):
    """
    XGBoost 모델 학습 및 반환
    :param X_train: 학습용 입력 데이터 (스케일링 전)
    :param y_train: 학습용 타겟 레이블
    :return: 학습된 XGBoost 모델
    """
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb_model.fit(X_train, y_train)
    return xgb_model

