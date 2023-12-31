import os
from functools import lru_cache

from attr import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    s3_location = os.getenv("S3_LOCATION", "")
    execution_role = os.getenv("ROLE", "")
    base_filepath = os.getenv("BASE_FILEPATH", "")
    data_filepath = os.getenv("DATA_FILEPATH", "")
    use_tuning_step = os.getenv("USE_TUNING_STEP", "")
    data_quality_location = os.getenv("DATA_QUALITY_LOCATION", "")
    model_package_group_name = "penguins"

    # scripts
    preprocessor_script_path = "src/steps/preprocessing.py"
    train_script_path = "src/steps/training.py"
    evaluate_script_path = "src/steps/evaluate.py"
    lambda_script_path = "src/steps/lambda.py"
    endpoint_code_path = "src/steps/inference"


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    return settings
