create_model_step = ModelStep(
    name="create",
    display_name="create-model",
    step_args=model.create(instance_type="ml.m5.large"),
)

transformer = Transformer(
    model_name=create_model_step.properties.ModelName,
    base_transform_job_name="transform",
    instance_type="ml.c5.xlarge",
    instance_count=1,
    accept="application/json",
    strategy="SingleRecord",
    assemble_with="Line",
    output_path=f"{S3_LOCATION}/transform",
    sagemaker_session=pipeline_session,
)

generate_test_predictions_step = TransformStep(
    name="generate-test-predictions",
    step_args=transformer.transform(
        # We will use the test dataset we generated during the preprocessing
        # step to run it through the model and generate predictions.
        data=preprocess_data_step.properties.ProcessingOutputConfig.Outputs[
            "test-baseline"
        ].S3Output.S3Uri,
        join_source="Input",
        content_type="application/json",
        split_type="Line",
    ),
    cache_config=cache_config,
)
