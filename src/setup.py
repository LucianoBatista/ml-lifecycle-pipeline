from sagemaker.workflow.pipeline import Pipeline

from pipelines.pipe import Pipe


def run():
    pipe_obj = Pipe()
    # initiating all the steps
    process_data_step = pipe_obj.preprocessing(
        job_name="penguins-preprocess-data-step",
        framework_version="0.23-1",
        instance_type="ml.t3.medium",
        instance_count=1,
        step_name="PreprocessDataStep",
    )
    data_quality_baseline_step = pipe_obj.data_quality(process_data_step)
    tune_model_step = pipe_obj.tuning(process_data_step)
    train_model_step = pipe_obj.training(process_data_step)
    evaluate_model_step = pipe_obj.evaluation(
        process_data_step, train_model_step, tune_model_step
    )
    create_model_step = pipe_obj.create_model_step(
        process_data_step, tune_model_step, train_model_step
    )
    generate_test_data_step = pipe_obj.test_predictions(
        process_data_step, create_model_step
    )
    model_quality_step = pipe_obj.model_quality(
        generate_test_data_step,
    )
    register_model_step = pipe_obj.register(
        data_quality_baseline_step,
        model_quality_step,
        train_model_step,
        tune_model_step,
    )
    deploy = pipe_obj.deploy(register_model_step)

    condition_step = pipe_obj.condition(
        evaluate_model_step=evaluate_model_step,
        create_model_step=create_model_step,
        generate_test_predictions_step=generate_test_data_step,
        model_quality_baseline_step=model_quality_step,
        register_model_step=register_model_step,
        deploy_step=deploy,
    )

    # Create a pipeline
    full_session_pipeline = Pipeline(
        name="penguins-session6-pipeline",
        parameters=[
            pipe_obj.dataset_location,
            pipe_obj.param_data_capture_percentage,
            pipe_obj.param_data_capture_destination,
            pipe_obj.param_accuracy_threshold,
        ],
        steps=[
            process_data_step,
            data_quality_baseline_step,
            tune_model_step if pipe_obj.use_tunning_step else train_model_step,
            evaluate_model_step,
            condition_step,
        ],
        pipeline_definition_config=pipe_obj.pipeline_definition_config,
        sagemaker_session=pipe_obj.sagemaker_session,
    )

    # Uploading the pipeline
    full_session_pipeline.upsert(role_arn=pipe_obj.role)

    # Starting the pipeline
    full_session_pipeline.start()
