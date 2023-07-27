from sagemaker.workflow.pipeline import Pipeline
from pipelines import preprocessing_setup, base_pipe_settings


def pipe_1(role: str):
    cache_config = base_pipe_settings.get_cache_config()
    dataset_location = base_pipe_settings.get_dataset_location_param()
    preprocessor_destination = base_pipe_settings.get_preprocessor_destination()
    pipeline_definition_config = base_pipe_settings.get_pipeline_definition()

    sklearn_processosr = preprocessing_setup.create_sklearn_processor(
        job_name="penguins-preprocess-data",
        framework_version="0.23-1",
        instance_type="ml.t3.medium",
        instance_count=1,
        role=role,
    )

    processing_step = preprocessing_setup.create_preprocessing_step(
        step_name="penguins-preprocess-data",
        preprocessor=sklearn_processosr,
        cache_config=cache_config,
        dataset_location=dataset_location,
        preprocessor_destination=preprocessor_destination,
        script_path="src/steps/preprocessing.py",
    )

    session1_pipeline = Pipeline(
        name="penguins-session1-pipeline",
        parameters=[
            dataset_location,
            preprocessor_destination,
        ],
        steps=[
            processing_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
    )
    return session1_pipeline


def pipe_2(
    role: str,
    dataset_location,
    preprocessor_destination,
    pipeline_definition_config,
    preprocess_data_step,
    train_model_step,
    tune_model_step,
    USE_TUNING_STEP: bool,
):
    session2_pipeline = Pipeline(
        name="penguins-session2-pipeline",
        parameters=[
            dataset_location,
            preprocessor_destination,
        ],
        steps=[
            preprocess_data_step,
            tune_model_step if USE_TUNING_STEP else train_model_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
    )

    session2_pipeline.upsert(role_arn=role)
