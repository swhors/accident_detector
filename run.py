import pandas as pd
import numpy as np

class AISafetyFinalSystem:
    def __init__(self, base_models, meta_model, scaler):
        """
        base_models: {'XGB': ..., 'RF': ..., 'SVM': ..., 'LSTM': ...}
        meta_model: 최종 가중치 결정용 Logistic Regression 모델
        scaler: 데이터 표준화 도구
        """
        self.models = base_models
        self.meta_model = meta_model
        self.scaler = scaler

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
        p1 = self.models['XGB'].predict_proba(input_data)[:, 1]
        p2 = self.models['RF'].predict_proba(input_data)[:, 1]
        p3 = self.models['SVM'].predict_proba(scaled)[:, 1]
        p4 = self.models['LSTM'].predict(lstm_in).flatten()
        
        # 2. 메타 모델(Stacking)을 통한 최종 결합
        meta_features = np.column_stack([p1, p2, p3, p4])
        final_prob = self.meta_model.predict_proba(meta_features)[:, 1]
        
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

# --- 시스템 가동 (예시) ---

# 1. 시스템 초기화 (기학습된 모델들 탑재)
safety_system = AISafetyFinalSystem(
    base_models={'XGB': model_xgb, 'RF': model_rf, 'SVM': model_svm, 'LSTM': model_lstm},
    meta_model=meta_model,
    scaler=scaler
)

# 2. 오늘 아침 'A 현장' 상황 데이터 유입 (샘플 데이터)
today_data = X_test.iloc[0:1] 

# 3. 통합 AI 분석 실시
final_risk, detail_probs = safety_system.predict_integrated_risk(today_data)

# 4. 방지 대책 생성 및 결과 출력
action_plan = safety_system.generate_action_plan(final_risk[0], today_data.iloc[0])

print(f"==========================================")
print(f"   [AI 안전 예견 시스템 분석 결과]   ")
print(f"==========================================")
print(f"▶ 최종 위험 점수: {final_risk[0]*100:.1f}점")
print(f"▶ 위험 등급: {action_plan['level']} ({action_plan['color']})")
print(f"▶ 모델별 의견: XGB({detail_probs['XGB'][0]:.2f}), LSTM({detail_probs['LSTM'][0]:.2f})")
print(f"\n[실천 지침]")
for action in action_plan['actions']:
    print(f" - {action}")
print(f"==========================================")

