from functools import lru_cache
from attr import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    s3_location = os.getenv("S3_LOCATION")
    execution_role = os.getenv("ROLE")
    base_filepath = os.getenv("BASE_FILEPATH")
    data_filepath = os.getenv("DATA_FILEPATH")


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    return settings
