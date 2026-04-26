import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def pre_process(df_final):
    # 데이터 인코딩 (날짜 제외, 범주형 변수 변환)
    df_ml = pd.get_dummies(df_final.drop(['date', 'site_name'], axis=1))
    X = df_ml.drop('accident_type', axis=1) if 'accident_type' in df_ml else df_ml
    y = LabelEncoder().fit_transform(df_final['accident_type']) # 사고 종류를 숫자로 변환

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 스케일링 (SVM과 LSTM은 필수)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test
