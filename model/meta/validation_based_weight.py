"""
Validation-based Weights

성능 기반 자동 가중치 (Validation-based Weights) :
모델을 학습시킨 후, 검증 데이터(Validation Set)에서 각 모델이
거둔 성적(예: F1-score 또는 정확도)을 그대로 가중치로 사용하는
방식입니다.
예) XGBoost가 0.9, SVM이 0.7의 점수를 받았다면, 점수 비율대로
    가중치를 배분합니다.
"""

from sklearn.metrics import f1_score


class ValidationBasedWeight():
    """성능 기반 자동 가중치 클래스
    - 각 모델의 검증 데이터 성적을 기반으로 가중치를 계산하여,
      최종 위험도 계산에 활용합니다.
    """
    def __init__(self, model_xgb, model_rf, model_svm, model_lstm):
        """초기화 및 모델 탑재
        :param model_xgb: 학습된 XGBoost 모델
        :param model_rf: 학습된 랜덤 포레스트 모델
        :param model_svm: 학습된 SVM 모델
        :param model_lstm: 학습된 LSTM 모델
        """
        super().__init__(model_xgb, model_rf, model_svm, model_lstm)
        self._weight = None
        self._total_score = None

    def train(self, X_val, X_val_scaled, X_val_lstm, y_val, verbose=0):
        """
        메타 모델(최종 결정권자) 학습
        :param X_val: 검증용 원본 데이터 (스케일링 전)
        :param X_val_scaled: 검증용 스케일링된 데이터 (SVM용)
        :param X_val_lstm: 검증용 LSTM 입력 데이터 (3차원)
        :param y_val: 검증용 타겟 레이블
        :param verbose: 학습 과정 출력 여부 (0: 출력 안함, 1: 출력)
        :return: 학습된 메타 모델
        """
        # 각 모델의 검증 데이터 성적(F1-score) 측정
        scores = {
            'XGB': f1_score(y_val, self._model_xgb.predict(X_val)),
            'RF': f1_score(y_val, self._model_rf.predict(X_val)),
            'SVM': f1_score(y_val, self._model_svm.predict(X_val_scaled)),
            'LSTM': f1_score(y_val, (self._model_lstm.predict(X_val_lstm) > 0.5).astype(int))
        }

        # 점수 총합으로 나누어 가중치(합계=1) 계산
        self._total_score = sum(scores.values())
        self._weights = {m: s / self._total_score for m, s in scores.items()}

        if verbose > 0:
            print("--- [성능 기반 가중치] ---")
            for m, w in self._weights.items():
                print(f"{m} 모델 가중치: {w:.4f}")

    # 3. 최종 위험도 계산 함수
    def manual_weighted_risk(self, probs):
        """최종 위험도 계산
        :param probs: 각 모델의 예측 확률값이 담긴 딕셔리 (예: {'XGB': 0.8, 'RF': 0.7, ...})
        :return: 최종 위험도 점수 (0~1 사이)
        """
        # probs = {'XGB': 0.8, 'RF': 0.7, ...}
        return sum(probs[m] * self._weights[m] for m in self._weights)
