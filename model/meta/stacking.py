"""
automatic weight based on stacking

스태킹(Stacking) 기반 자동 가중치 학습법 :
이 방식은 '최종 결정 AI 모델'을 하나 더 두어, 각 모델의 예측 결과를
보고 최적의 조합을 스스로 찾아냅니다.
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
from model.meta.weight import Weight


class Stacking(Weight):
    """스태킹(Stacking) 모델 클래스
    - 각 모델의 예측 결과를 모아서 최종 위험도를 예측하는 메타 모델입니다.
    - 메타 모델로는 Logistic Regression을 사용하여, 각 모델의 예측값에 대한 최적의 가중치를 학습합니다.
    """
    def _create_meta_features(self, X_data, X_scaled, X_lstm):
        """각 모델의 예측 확률을 모아서 메타 모델의 입력 데이터셋 생성
        :param X_data: 원본 입력 데이터 (스케일링 전)
        :param X_scaled: 스케일링된 입력 데이터 (SVM용)
        :param X_lstm: LSTM용 3차원 입력 데이터
        :return: 메타 모델의 입력으로 사용할 데이터셋 (각 모델의 예측 확률값을 컬럼으로 갖는 2차원 배열)
        """
        # 메타 데이터 생성 (각 모델의 예측 확률을 하나의 데이터셋으로 합침)
        xgb_prob = self._model_xgb.predict_proba(X_data)[:, 1]
        rf_prob = self._model_rf.predict_proba(X_data)[: , 1]
        svm_prob = self._model_svm.predict_proba(X_scaled)[:, 1]
        lstm_prob = self._model_lstm.predict(X_lstm)#.flatten()
        return np.column_stack([
            xgb_prob,
            rf_prob,
            svm_prob,
            lstm_prob
        ])

    def train(self, X_val, X_val_scaled, X_val_lstm, y_val, verbose=1):
        """메타 모델(최종 결정권자) 학습 및 최적 가중치 도출
        :param X_val: 검증용 원본 데이터 (스케일링 전)
        :param X_val_scaled: 검증용 스케일링된 데이터 (SVM용)
        :param X_val_lstm: 검증용 LSTM 입력 데이터 (3차원)
        :param y_val: 검증용 타겟 레이블
        :param verbose: 학습 과정 출력 여부 (0: 출력 안함, 1: 출력)
        :return: 학습된 메타 모델
        """
        # 검증 데이터로 메타 모델 학습
        train_meta = self._create_meta_features(X_val, X_val_scaled, X_val_lstm)
        self._meta_model = LogisticRegression()
        self._meta_model.fit(train_meta, y_val)
        if verbose > 0:
            print("\n--- [스태킹 모델 분석] ---")
            print("데이터가 스스로 판단한 모델별 영향력(Coefficients):")
            for name, coef in zip(['XGB', 'RF', 'SVM', 'LSTM'], self._meta_model.coef_[0]):
                print(f"{name}: {coef:.4f}")
        return self._meta_model
        

    def predict(self, X_new, X_new_scaled, X_new_lstm):
        """최종 위험도 예측
        :param X_new: 새로운 입력 데이터 (스케일링 전)
        :param X_new_scaled: 새로운 입력 데이터 (스케일링된, SVM용)
        :param X_new_lstm: 새로운 입력 데이터 (LSTM용 3차원)
        :return: 최종 위험도 예측값 (0~1 사이의 확률)
        """
        # 최종 위험도 예측 함수
        meta_feat = self._create_meta_features(X_new, X_new_scaled, X_new_lstm)
        # 0~1 사이의 최종 위험 확률 반환
        return self._meta_model.predict_proba(meta_feat)[:, 1]


    def predict(self, inputdata):
        """최종 위험도 예측
        :param args: 각 모델의 입력 데이터 (inputdata)
        """
        print(f'inputdata: {inputdata}')  # Debugging line to check the input data
        return self._meta_model.predict_proba(inputdata)[:, 1]

    @staticmethod
    def create_model(models: list,
                     X_val, X_val_scaled, X_val_lstm, y_val, verbose=1):
        """스태킹 모델 생성
        :param models: 각 모델이 담긴 리스트 (예: [model_xgb, model_rf, model_svm, model_lstm])
        :param X_val: 검증용 원본 데이터 (스케일링 전)
        :param X_val_scaled: 검증용 스케일링된 데이터 (SVM용)
        :param X_val_lstm: 검증용 LSTM 입력 데이터 (3차원)
        :param y_val: 검증용 타겟 레이블
        :param verbose: 학습 과정 출력 여부 (0: 출력 안함, 1: 출력)
        :return: 학습된 스태킹 모델
        """
        stacking = Stacking(*models)
        stacking.train(X_val, X_val_scaled, X_val_lstm, y_val, verbose)
        return stacking
