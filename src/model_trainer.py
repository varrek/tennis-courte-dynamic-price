import joblib
from sklearn.ensemble import RandomForestRegressor
import shap
import os

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.explainer = None
        self.preprocessor = None
        
    def train(self, X, y, preprocessor):
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)
        self.preprocessor = preprocessor
        
    def save_model(self, path='models/trained_model.joblib'):
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'explainer': self.explainer,
            'preprocessor': self.preprocessor
        }
        joblib.dump(model_data, path)
        
    def load_model(self, path='models/trained_model.joblib'):
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.explainer = model_data['explainer']
        self.preprocessor = model_data['preprocessor']
        return self.preprocessor
        
    def predict(self, X):
        return self.model.predict(X)
        
    def explain_prediction(self, X):
        shap_values = self.explainer.shap_values(X)
        return shap_values 