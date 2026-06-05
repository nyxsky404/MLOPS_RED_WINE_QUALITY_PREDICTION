import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        # Don't load the model here anymore
        self.model = None

    def predict(self, data):
        # Load the model only when we actually need to predict
        if self.model is None:
            model_path = Path('artifacts/model_trainer/model.joblib')
            self.model = joblib.load(model_path)
        
        prediction = self.model.predict(data)
        return prediction