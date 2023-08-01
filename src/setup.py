from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from pipelines import (
    base_pipe_settings,
    checks,
    evaluate,
    preprocessing_setup,
    registration,
    training,
    tuning,
)


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


def pipe_3(role: str, USE_TUNING_STEP: bool = True):
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

    # evaluation step
    evaluation_step, evaluation_report = evaluate.create_evalutaion_step(
        tensorflow_processor=tensorflow_estimator,
        preprocess_data_step=processing_step,
        train_model_step=train_or_tune_step,
        tune_model_step=train_or_tune_step,
        sagemaker_session=sagemaker_session,
        script_path="src/steps/evaluate.py",
        s3_location=S3_BUCKET,
        cache_config=cache_config,
    )

    model_metrics = evaluate.create_model_metrics(
        evaluation_step=evaluation_step,
    )

    # register step
    register_model_step = registration.create_registration_step(
        pipeline_session=sagemaker_session,
        role=role,
        train_model_step=train_or_tune_step,
        tune_model_step=train_or_tune_step,
        model_package_group_name="penguins-model-package-group",
        model_metrics=model_metrics,
        sagemaker_session=sagemaker_session,
    )

    # condition step
    condition_step, accuracy_threshold = checks.create_condition_step(
        evaluate_model_step=evaluation_step,
        evaluation_report=evaluation_report,
        register_model_step=register_model_step,
    )

    # pipeline
    session3_pipeline = Pipeline(
        name="penguins-session3-pipeline",
        parameters=[
            dataset_location,
            accuracy_threshold,
        ],
        steps=[
            processing_step,
            train_or_tune_step,
            evaluation_step,
            condition_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
        sagemaker_session=sagemaker_session,
    )

    session3_pipeline.upsert(role_arn=role)


def pipe_4(role: str):
    session4_pipeline = Pipeline(
        name="penguins-session4-pipeline",
        parameters=[
            dataset_location,
            accuracy_threshold,
            data_capture_percentage,
            data_capture_destination,
        ],
        steps=[
            preprocess_data_step,
            tune_model_step if USE_TUNING_STEP else train_model_step,
            evaluate_model_step,
            condition_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
        sagemaker_session=pipeline_session,
    )

    session4_pipeline.upsert(role_arn=role)


def pipe_5(role: str):
    session5_pipeline = Pipeline(
        name="penguins-session5-pipeline",
        parameters=[
            dataset_location,
            data_capture_percentage,
            data_capture_destination,
            accuracy_threshold,
        ],
        steps=[
            preprocess_data_step,
            data_quality_baseline_step,
            tune_model_step if USE_TUNING_STEP else train_model_step,
            evaluate_model_step,
            condition_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
        sagemaker_session=pipeline_session,
    )

    session5_pipeline.upsert(role_arn=role)


def pipe_6(role: str):
    session6_pipeline = Pipeline(
        name="penguins-session6-pipeline",
        parameters=[
            dataset_location,
            data_capture_percentage,
            data_capture_destination,
            accuracy_threshold,
        ],
        steps=[
            preprocess_data_step,
            data_quality_baseline_step,
            tune_model_step if USE_TUNING_STEP else train_model_step,
            evaluate_model_step,
            condition_step,
        ],
        pipeline_definition_config=pipeline_definition_config,
        sagemaker_session=pipeline_session,
    )

    session6_pipeline.upsert(role_arn=role)
