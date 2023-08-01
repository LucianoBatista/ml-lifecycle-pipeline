from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.functions import Join
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep

USE_TUNING_STEP = False


def create_evalutaion_step(
    tensorflow_processor,
    preprocess_data_step,
    train_model_step,
    tune_model_step,
    sagemaker_session,
    script_path,
    s3_location,
    cache_config,
):
    evaluation_report = PropertyFile(
        name="evaluation-report",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluate_model_step = ProcessingStep(
        name="evaluate-model",
        step_args=tensorflow_processor.run(
            inputs=[
                ProcessingInput(
                    source=preprocess_data_step.properties.ProcessingOutputConfig.Outputs[
                        "test"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                ),
                ProcessingInput(
                    source=(
                        tune_model_step.get_top_model_s3_uri(
                            top_k=0, s3_bucket=sagemaker_session.default_bucket()
                        )
                        if USE_TUNING_STEP
                        else train_model_step.properties.ModelArtifacts.S3ModelArtifacts
                    ),
                    destination="/opt/ml/processing/model",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=f"{s3_location}/evaluation",
                ),
            ],
            code=script_path,
        ),
        property_files=[evaluation_report],
        cache_config=cache_config,
    )

    return evaluate_model_step, evaluation_report


def create_model_metrics(evaluation_step):
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0][
                        "S3Output"
                    ]["S3Uri"],
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )
    return model_metrics
