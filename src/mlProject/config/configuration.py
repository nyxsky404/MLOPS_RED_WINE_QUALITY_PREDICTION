from pathlib import Path
from mlProject.constants import *
from mlProject import logger
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (DataIngestionConfig,
                                            DataValidationConfig,
                                            DataTransformationConfig,
                                            ModelTrainerConfig,
                                            ModelEvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def _validate_config_keys(self, config_section: dict, required_keys: list, section_name: str):
        missing = [key for key in required_keys if key not in config_section]
        if missing:
            raise KeyError(
                f"Missing required config keys in '{section_name}': {missing}"
            )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        self._validate_config_keys(config, ["root_dir", "source_URL", "local_data_file", "unzip_dir"], "data_ingestion")

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir) 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        self._validate_config_keys(config, ["root_dir", "STATUS_FILE", "data_file"], "data_validation")

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=Path(config.STATUS_FILE),
            data_file = Path(config.data_file),
            all_schema=schema,
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.DataTransformation
        self._validate_config_keys(config, ["root_dir", "data_path"], "data_transformation")

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            test_size=params.test_size,
            random_state=params.random_state,
            stratify_column=params.stratify_column,
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN
        self._validate_config_keys(config, ["root_dir", "train_data_path", "test_data_path", "model_name"], "model_trainer")

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_data_path = Path(config.train_data_path),
            test_data_path = Path(config.test_data_path),
            model_name = config.model_name,
            alpha = params.alpha,
            l1_ratio = params.l1_ratio,
            target_column = schema.name
            
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN
        self._validate_config_keys(config, ["root_dir", "test_data_path", "model_path", "metric_file_name"], "model_evaluation")

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path = Path(config.model_path),
            all_params=params,
            metric_file_name = Path(config.metric_file_name),
            target_column = schema.name
           
        )

        return model_evaluation_config
