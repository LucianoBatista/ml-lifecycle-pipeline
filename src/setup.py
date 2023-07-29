from sagemaker.workflow.pipeline import Pipeline
from pipelines import preprocessing_setup, base_pipe_settings, training, tuning
from sagemaker.workflow.pipeline_context import PipelineSession


# For now, we're going to have a lot of duplication here, because the ideia
# is to have a pipeline for each session. In the future, we can have a
# unique pipeline with all the steps and then remove all duplication
def pipe_1(role: str):
    cache_config = base_pipe_settings.get_cache_config()
    dataset_location = base_pipe_settings.get_dataset_location_param()
    preprocessor_destination = base_pipe_settings.get_preprocessor_destination()
    pipeline_definition_config = base_pipe_settings.get_pipeline_definition()
    _ = PipelineSession()

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
    USE_TUNING_STEP: bool = True,
):
    # base config
    dataset_location = base_pipe_settings.get_dataset_location_param()
    preprocessor_destination = base_pipe_settings.get_preprocessor_destination()
    pipeline_definition_config = base_pipe_settings.get_pipeline_definition()
    cache_config = base_pipe_settings.get_cache_config()
    sagemaker_session = PipelineSession()

    # preprocessing step
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

    # training and tuning step
    tensorflow_estimator = training.create_tensorflow_estimator(
        role, "src/steps/training.py", sagemaker_session
    )

    if USE_TUNING_STEP:
        tuner = tuning.create_tuner(tensorflow_estimator)
        train_or_tune_step = tuning.create_tuner_step(
            preprocess_data_step=processing_step,
            tuner=tuner,
            cache_config=cache_config,
        )
    else:
        train_or_tune_step = training.create_training_step(
            estimator=tensorflow_estimator,
            preprocess_data_step=processing_step,
            cache_config=cache_config,
        )

    # pipeline
    session2_pipeline = Pipeline(
        name="penguins-session2-pipeline",
        parameters=[
            dataset_location,
            preprocessor_destination,
        ],
        steps=[
            processing_step,
            train_or_tune_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
    )

    return session2_pipeline
