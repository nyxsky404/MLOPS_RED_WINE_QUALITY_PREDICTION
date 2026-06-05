import os
import joblib
from mlProject import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig


NUMERIC_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
]


class FeatureScaler:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
        ])

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        scaled = self.pipeline.fit_transform(X[NUMERIC_FEATURES])
        result = X.copy()
        result[NUMERIC_FEATURES] = scaled
        joblib.dump(self.pipeline, self.save_path)
        logger.info(f"Feature scaler saved to {self.save_path}")
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        scaled = self.pipeline.transform(X[NUMERIC_FEATURES])
        result = X.copy()
        result[NUMERIC_FEATURES] = scaled
        return result


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        try:
            data = pd.read_csv(self.config.data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.config.data_path}")
            raise
        except Exception as e:
            logger.exception(f"Failed to read data file: {self.config.data_path}")
            raise

        stratify = None
        if self.config.stratify_column:
            if self.config.stratify_column not in data.columns:
                raise ValueError(
                    f"Stratify column '{self.config.stratify_column}' "
                    "not found in transformed data"
                )
            stratify = data[self.config.stratify_column]

        try:
            train, test = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify,
            )
        except ValueError as exc:
            if stratify is None:
                logger.exception("train_test_split failed")
                raise
            logger.warning(
                "Falling back to non-stratified split because '%s' cannot be "
                "stratified safely: %s",
                self.config.stratify_column,
                exc,
            )
            train, test = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

        scaler_save_path = os.path.join(self.config.root_dir, "scaler.joblib")
        scaler = FeatureScaler(scaler_save_path)
        train = scaler.fit_transform(train)
        test = scaler.transform(test)

        try:
            train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        except OSError as e:
            logger.error(f"Failed to write train/test CSV files: {e}")
            raise

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
