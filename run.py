import pandas as pd
import numpy as np

from data.datagen import gen_data
from data.preprocess import pre_process
from model.lstm import train_lstm
from model.randomforest import train_randomforest
from model.svm import train_svm
from model.xgboost import train_xgboost
from model.ai_safety_system import AISafetySystem
from model.stacking import Stacking
from sklearn.preprocessing import StandardScaler


df_final = gen_data(None)

X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test = pre_process(df_final=df_final)


model_xgb = train_xgboost(X_train=X_train, y_train=y_train)
model_rf = train_randomforest(X_train=X_train, y_train=y_train)
model_svm = train_svm(X_train_scaled=X_test_scaled, y_train=y_train)
model_lstm = train_lstm(X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, y_train=y_train)

meta_model = Stacking(model_lstm=model_lstm, model_rf=model_rf, model_svm=model_svm, model_xgb=model_xgb)

scaler = StandardScaler()

# 1. 시스템 초기화 (기학습된 모델들 탑재)
safety_system = AISafetySystem(
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

