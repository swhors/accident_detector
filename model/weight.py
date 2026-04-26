"""
weight.py

"""

class Weight():
    def __init__(self, model_xgb, model_rf, model_svm, model_lstm):
        self._model_xgb = model_xgb
        self._model_rf = model_rf
        self._model_svm = model_svm
        self._model_lstm = model_lstm
        self._meta_model = None
    
    def train(self, X_val, X_val_scaled, X_val_lstm, y_val, verbose=0):
        pass
    
    def predict(self, X_new, X_new_scaled, X_new_lstm):
        pass
