from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np


def train_lstm(X_train_scaled, X_test_scaled, y_train):
    """
    LSTM 모델 학습 및 반환
    :param X_train_scaled: LSTM 학습용 스케일링된 입력 데이터 (2차원)
    :param X_test_scaled: LSTM 테스트용 스케일링된 입력 데이터 (2차원)
    :param y_train: 학습용 타겟 레이블
    :return: 학습된 LSTM 모델
    """
    # LSTM을 위한 3차원 변환 (샘플 수, 타임스텝, 특성 수)
    X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    # X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    model_lstm = Sequential([
        LSTM(64, input_shape=(1, X_train_scaled.shape[1])),
        Dense(32, activation='relu'),
        # Dense(len(np.unique(y_train)), activation='softmax') # 클래스 수만큼 출력
        Dense(1, activation='sigmoid')  # 이진 분류용 출력
    ])

    # model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_lstm.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)
    # lstm_pred = np.argmax(model_lstm.predict(X_test_lstm), axis=1)
    
    return model_lstm
