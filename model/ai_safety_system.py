import numpy as np
import pandas as pd

class AISafetySystem:
    def __init__(self, base_models, meta_model, scaler):
        """
        base_models: {'XGB': ..., 'RF': ..., 'SVM': ..., 'LSTM': ...}
        meta_model: 최종 가중치 결정용 Logistic Regression 모델
        scaler: 데이터 표준화 도구
        """
        self._models = base_models
        self._meta_model = meta_model
        self._scaler = scaler

    def _process_data(self, input_data):
        """실시간 입력 데이터를 각 모델 규격에 맞게 변환"""
        # 기본 스케일링
        scaled = self.scaler.transform(input_data)
        # LSTM용 3차원 변환
        lstm_in = np.reshape(scaled, (scaled.shape[0], 1, scaled.shape[1]))
        return scaled, lstm_in

    def predict_integrated_risk(self, input_data):
        """4개 모델의 결과를 스태킹 모델로 통합하여 최종 위험도 산출"""
        scaled, lstm_in = self._process_data(input_data)
        
        # 1. 개별 모델 예측 (확률값)
        p1 = self._models['XGB'].predict_proba(input_data)[:, 1]
        p2 = self._models['RF'].predict_proba(input_data)[:, 1]
        p3 = self._models['SVM'].predict_proba(scaled)[:, 1]
        p4 = self._models['LSTM'].predict(lstm_in).flatten()
        
        # 2. 메타 모델(Stacking)을 통한 최종 결합
        meta_features = np.column_stack([p1, p2, p3, p4])
        final_prob = self._meta_model.predict(meta_features)[:, 1]
        
        return final_prob, {'XGB': p1, 'RF': p2, 'SVM': p3, 'LSTM': p4}

    def generate_action_plan(self, risk_prob, row_data):
        """위험도와 현장 상황을 매칭하여 구체적인 예방 지침 하달"""
        score = risk_prob * 100
        plan = {"level": "", "color": "", "actions": []}

        # 위험 등급 설정
        if score >= 80:
            plan.update({"level": "심각(CRITICAL)", "color": "RED"})
            plan["actions"].append("!!! 즉시 전 공정 작업 중지 및 근로자 대피 !!!")
        elif score >= 50:
            plan.update({"level": "경계(WARNING)", "color": "ORANGE"})
            plan["actions"].append("안전관리자 현장 상주 및 위험 구역 통제 강화")
        else:
            plan.update({"level": "보통(NORMAL)", "color": "GREEN"})
            plan["actions"].append("개인 보호구 착용 점검 및 일상적 안전 수칙 준수")

        # 데이터 기반 정밀 대응 지침 (예시)
        if row_data['is_rainy'] == 1:
            plan["actions"].append("비 내림: 고소 작업(비계, 지붕) 전면 금지 및 미끄럼 주의보 발령")
        if row_data['worker_type_비상시'] == 1:
            plan["actions"].append("비상시 근로자 감지: 투입 전 특별 안전교육 30분 실시 필수")
            
        return plan