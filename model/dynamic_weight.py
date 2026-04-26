"""
Dynamic Weight

"""
from sklearn.linear_model import LogisticRegression
from model.weight import Weight


class DynamicWeight(Weight):
    def train(self, X_val, X_val_scaled, X_val_lstm, y_val, verbose=0):
        # 각 모델의 예측값(확률)을 모음
        # train_meta의 형태: [XGB_prob, RF_prob, SVM_prob, LSTM_prob]
        self._train_meta = np.column_stack([
            self._model_xgb.predict_proba(X_val)[:, 1],
            self._model_rf.predict_proba(X_val)[:, 1],
            self._model_svm.predict_proba(X_val_scaled)[:, 1],
            self._model_lstm.predict(X_val_lstm)
        ])
        # 메타 모델(최종 결정권자) 학습
        # 이 모델이 사실상 '가중치 결정기' 역할을 합니다.
        self._meta_model = LogisticRegression()
        self._meta_model.fit(self._train_meta, y_val)

        if verbose > 0:
            # 최종 가중치 확인
            # 각 모델에 부여된 계수(Coefficient)가 곧 데이터가 찾아낸 최적 가중치입니다.
            print("AI가 결정한 모델별 중요도(가중치):", self._meta_model.coef_)
        return self._meta_model
