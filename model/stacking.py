"""
automatic weight based on stacking

스태킹(Stacking) 기반 자동 가중치 학습법 :
이 방식은 '최종 결정 AI 모델'을 하나 더 두어, 각 모델의 예측 결과를
보고 최적의 조합을 스스로 찾아냅니다.
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
from model.weight import Weight


class Stacking(Weight):
    def _create_meta_features(self, X_data, X_scaled, X_lstm):
        # 메타 데이터 생성 (각 모델의 예측 확률을 하나의 데이터셋으로 합침)
        return np.column_stack([
            self._model_xgb.predict_proba(X_data)[:, 1],
            self._model_rf.predict_proba(X_data)[:, 1],
            self._model_svm.predict_proba(X_scaled)[:, 1],
            self._model_lstm.predict(X_lstm).flatten()
        ])

    def train(self, X_val, X_val_scaled, X_val_lstm, y_val, verbose=1):
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
        # 최종 위험도 예측 함수
        meta_feat = self._create_meta_features(X_new, X_new_scaled, X_new_lstm)
        # 0~1 사이의 최종 위험 확률 반환
        return self._meta_model.predict_proba(meta_feat)[:, 1]
