import json
import tempfile
import pandas as pd
import os
import numpy as np
from mlProject import logger
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject.utils.model_registry import (
    get_version_id, compute_file_hash,
)
from mlProject.components.data_transformation import NUMERIC_FEATURES
from mlProject.utils.mlflow_tracker import MlflowTracker


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.mlflow_tracker = None

    
    def train(self):
        try:
            train_data = pd.read_csv(self.config.train_data_path)
        except FileNotFoundError as e:
            logger.error(f"Training data file not found: {e.filename}")
            raise
        except Exception as e:
            logger.exception("Failed to load training data")
            raise

        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]

        # Load preprocessor if available (from data_transformation stage)
        preprocessor = None
        preprocessor_path = self.config.preprocessor_path or Path('artifacts/data_transformation/preprocessor.joblib')
        if self.config.use_scaler and preprocessor_path.exists():
            try:
                preprocessor = joblib.load(preprocessor_path)
                logger.info(f"Loaded preprocessor from {preprocessor_path}")
            except Exception as e:
                logger.warning(f"Failed to load preprocessor: {e}. Training model without preprocessor.")

        # Create unified pipeline: preprocessor + model
        if preprocessor is not None:
            expected_cols = len(NUMERIC_FEATURES)
            if train_x.shape[1] != expected_cols:
                logger.warning(
                    f"train_x has {train_x.shape[1]} columns but preprocessor "
                    f"expects {expected_cols}. Selecting NUMERIC_FEATURES."
                )
                train_x = train_x[NUMERIC_FEATURES]

            train_x_preprocessed = preprocessor.transform(train_x)

            if train_x_preprocessed.shape[1] <= train_x.shape[1]:
                logger.warning(
                    f"Preprocessor output dimension {train_x_preprocessed.shape[1]} "
                    f"is not larger than input {train_x.shape[1]} — verify pipeline"
                )

            # Train model on transformed data
            try:
                lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
                lr.fit(train_x_preprocessed, train_y)
            except Exception as e:
                logger.exception("Failed to train model")
                raise

            # Create unified pipeline for inference
            unified_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", lr),
            ])
            logger.info("Created unified pipeline: preprocessor + model")
        else:
            # Train model directly on raw data if no preprocessor
            train_x_preprocessed = train_x
            try:
                lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
                lr.fit(train_x_preprocessed, train_y)
                unified_pipeline = lr
            except Exception as e:
                logger.exception("Failed to train model")
                raise

        self._init_mlflow()

        version_id = get_version_id()
        model_filename = f"model_{version_id}.joblib"
        model_path_str = os.path.join(self.config.root_dir, model_filename)
        try:
            with tempfile.NamedTemporaryFile(dir=self.config.root_dir, suffix='.joblib', delete=False) as tmp:
                tmp_path = tmp.name
                joblib.dump(unified_pipeline, tmp_path)
            os.replace(tmp_path, model_path_str)
            checksum_path = model_path_str + ".sha256"
            from mlProject.utils.common import save_checksum
            save_checksum(Path(model_path_str), Path(checksum_path))
        except Exception as e:
            logger.exception(f"Failed to save model to {model_path_str}")
            raise

        model_path = Path(model_path_str)
        data_hash = None
        try:
            data_hash = compute_file_hash(Path(self.config.train_data_path))
        except Exception as e:
            logger.warning(f"Could not compute data hash: {e}")

        params = {
            "alpha": self.config.alpha,
            "l1_ratio": self.config.l1_ratio,
        }

        model_info = {
            "version_id": version_id,
            "model_path": str(model_path),
            "params": params,
            "data_hash": data_hash or "",
        }
        model_info_path = os.path.join(self.config.root_dir, "model_info.json")
        with open(model_info_path, "w") as f:
            json.dump(model_info, f, indent=2)

        self._log_to_mlflow(unified_pipeline, version_id, model_path, params, train_x)

        logger.info(f"Unified pipeline (preprocessor + model) {version_id} trained and saved to {model_path}")
        logger.info(f"Train X shape: {train_x_preprocessed.shape}")

    def _init_mlflow(self):
        try:
            from mlProject.config.configuration import ConfigurationManager
            config_manager = ConfigurationManager()
            registry_config = config_manager.get_model_registry_config()
            if registry_config.use_mlflow:
                self.mlflow_tracker = MlflowTracker(
                    tracking_uri=registry_config.mlflow_tracking_uri,
                    experiment_name=registry_config.mlflow_experiment_name,
                    use_mlflow=True,
                    registry_uri=registry_config.mlflow_registry_uri or None,
                )
                if self.mlflow_tracker.start_run(run_name="train"):
                    self.mlflow_tracker.log_params({
                        "alpha": self.config.alpha,
                        "l1_ratio": self.config.l1_ratio,
                        "model_type": "ElasticNet",
                    })
                    logger.info("MLflow tracking initialized for training")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            self.mlflow_tracker = None

    def _log_to_mlflow(self, model, version_id, model_path, params, train_x):
        if not self.mlflow_tracker or not self.mlflow_tracker.active_run:
            return
        try:
            from mlflow.models import infer_signature
            signature = infer_signature(train_x, model.predict(train_x))
            self.mlflow_tracker.log_model(
                model=model,
                artifact_path="model",
                signature=signature,
                input_example=train_x.iloc[:5] if hasattr(train_x, 'iloc') else train_x[:5],
                registered_model_name=None,
            )
            self.mlflow_tracker.log_artifact(str(model_path), artifact_path="artifacts")
            preprocessor_path = self.config.preprocessor_path
            if preprocessor_path and Path(preprocessor_path).exists():
                self.mlflow_tracker.log_artifact(str(preprocessor_path), artifact_path="artifacts")
            logger.info(f"Model {version_id} logged to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log model to MLflow: {e}")
        finally:
            self.mlflow_tracker.end_run()

        
