from sagemaker.tensorflow import TensorFlowModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.steps import CacheConfig

from config.config import get_settings

USE_TUNING_STEP = False

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


def get_pipeline_definition():
    pipeline_definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)
    return pipeline_definition_config


def create_registration_step(model_package_group_name, model_metrics, model):
    register_model_step = ModelStep(
        name="register-model",
        step_args=model.register(
            model_package_group_name=model_package_group_name,
            model_metrics=model_metrics,
            approval_status="Approved",
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            domain="MACHINE_LEARNING",
            task="CLASSIFICATION",
            framework="TENSORFLOW",
            framework_version="2.6",
        ),
    )
    return register_model_step
