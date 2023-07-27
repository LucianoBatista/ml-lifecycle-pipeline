from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from config.config import get_settings

S3_LOCATION = get_settings().s3_location


def get_cache_config():
    cache_config = CacheConfig(enable_caching=True, expire_after="15d")
    return cache_config


def get_dataset_location_param():
    dataset_location = ParameterString(
        name="dataset_location",
        default_value=f"{S3_LOCATION}/data.csv",
    )
    return dataset_location


def get_preprocessor_destination():
    preprocessor_destination = ParameterString(
        name="preprocessor_destination",
        default_value=f"{S3_LOCATION}/preprocessing",
    )
    return preprocessor_destination


def get_pipeline_definition():
    pipeline_definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)
    return pipeline_definition_config
