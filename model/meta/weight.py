"""
weight.py

"""

class Weight():
    """가중치 클래스
    - 각 모델의 예측 결과를 조합하여 최종 위험도를 계산할 때,
      모델별로 다른 가중치를 부여하는 방식의 기본 클래스입니다.
    - 이 클래스는 단순히 모델별 가중치를 저장하고, 최종 위험도 계산에 활용하는
      역할을 합니다.
    - 실제 가중치 계산 방식은 이 클래스를 상속받는 다른 클래스에서 구현됩니다.
    - 예) ValidationBasedWeight, Stacking, DynamicWeight 등에서 구체적인
      가중치 계산 로직이 구현됩니다.
    """
    def __init__(self, model_xgb, model_rf, model_svm, model_lstm):
        """초기화 및 모델 탑재
        :param model_xgb: 학습된 XGBoost 모델
        :param model_rf: 학습된 랜덤 포레스트 모델
        :param model_svm: 학습된 SVM 모델
        :param model_lstm: 학습된 LSTM 모델
        """
        self._model_xgb = model_xgb
        self._model_rf = model_rf
        self._model_svm = model_svm
        self._model_lstm = model_lstm
        self._meta_model = None
    
    def train(self, X_val, X_val_scaled, X_val_lstm, y_val, verbose=0):
        """"메타 모델(최종 결정권자) 학습
        :param X_val: 검증용 원본 데이터 (스케일링 전)
        :param X_val_scaled: 검증용 스케일링된 데이터 (SVM용)
        :param X_val_lstm: 검증용 LSTM 입력 데이터 (3차원)
        :param y_val: 검증용 타겟 레이블
        :param verbose: 학습 과정 출력 여부 (0: 출력 안함, 1: 출력)
        :return: 학습된 메타 모델"""
        pass
    
    def predict(self, X_new, X_new_scaled, X_new_lstm):
        """최종 위험도 예측
        :param X_new: 새로운 입력 데이터 (스케일링 전)
        :param X_new_scaled: 새로운 입력 데이터 (스케일링된, SVM용)
        :param X_new_lstm: 새로운 입력 데이터 (LSTM용 3차원)
        :return: 최종 위험도 예측값 (0~1 사이의 확률)
        """
        pass

    def predict(self, inputdata):
        """최종 위험도 예측
        :param inputdata: 모델별 입력 데이터 (예: X_xgb, X_rf, X_svm, X_lstm)
        """
        pass
