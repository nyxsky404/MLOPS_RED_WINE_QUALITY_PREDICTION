import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from mlProject.utils.model_registry import get_production_model_path, load_registry

class PredictionPipeline:
    def __init__(self, use_production_alias: bool = True):
        self.model = None
        self._model_path = None
        self.use_production_alias = use_production_alias

    def _resolve_model_path(self):
        """Resolve model path using registry production alias."""
        registry_path = Path('artifacts/model_registry.json')
        if self.use_production_alias and registry_path.exists():
            prod_path = get_production_model_path(registry_path)
            if prod_path and prod_path.exists():
                return prod_path
        return Path('artifacts/model_trainer/model.joblib')

    def predict(self, data):
        if self.model is None:
            model_path = self._resolve_model_path()
            self._model_path = model_path
            self.model = joblib.load(model_path)
        
        prediction = self.model.predict(data)
        return prediction