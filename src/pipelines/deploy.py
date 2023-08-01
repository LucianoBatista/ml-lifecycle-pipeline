from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.workflow.functions import Join
from pathlib import Path

from sagemaker.workflow.model_step import ModelStep


USE_TUNING_STEP = False
ENDPOINT_CODE_FOLDER = Path("src/steps/endpoint")


def create_model_inference(
    pipeline_session,
    role,
    preprocess_data_step,
    train_model_step,
    tune_model_step,
    model_package_group_name,
    sagemaker_session,
    model_metrics,
):
    model = TensorFlowModel(
        name="penguins",
        model_data=(
            tune_model_step.get_top_model_s3_uri(
                top_k=0, s3_bucket=sagemaker_session.default_bucket()
            )
            if USE_TUNING_STEP
            else train_model_step.properties.ModelArtifacts.S3ModelArtifacts
        ),
        entry_point="inference.py",
        source_dir=str(ENDPOINT_CODE_FOLDER),
        env={
            "PIPELINE_S3_LOCATION": Join(
                on="/",
                values=[
                    preprocess_data_step.properties.ProcessingOutputConfig.Outputs[
                        "pipeline"
                    ].S3Output.S3Uri,
                    "pipeline.pkl",
                ],
            ),
            "CLASSES_S3_LOCATION": Join(
                on="/",
                values=[
                    preprocess_data_step.properties.ProcessingOutputConfig.Outputs[
                        "classes"
                    ].S3Output.S3Uri,
                    "classes.csv",
                ],
            ),
        },
        framework_version="2.6",
        sagemaker_session=pipeline_session,
        role=role,
    )

    register_model_step = ModelStep(
        name="register",
        display_name="register-model",
        step_args=model.register(
            model_package_group_name=model_package_group_name,
            model_metrics=model_metrics,
            approval_status="Approved",
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            domain="MACHINE_LEARNING",
            task="CLASSIFICATION",
            framework="TENSORFLOW",
            framework_version="2.6",
        ),
    )


def lambda_inference():
    data_capture_percentage = ParameterInteger(
        name="data_capture_percentage",
        default_value=100,
    )

    data_capture_destination = ParameterString(
        name="data_capture_destination",
        default_value=f"{S3_LOCATION}/monitoring/data-capture",
    )

    deploy_fn = Lambda(
        function_name="deploy_fn",
        execution_role_arn=lambda_role,
        script=str(CODE_FOLDER / "lambda.py"),
        handler="lambda.lambda_handler",
        timeout=600,
        session=pipeline_session,
    )

    deploy_fn.upsert()

    deploy_step = LambdaStep(
        name="deploy",
        lambda_func=deploy_fn,
        inputs={
            # We use the ARN of the model we registered to
            # deploy it to the endpoint.
            "model_package_arn": register_model_step.properties.ModelPackageArn,
            "endpoint_name": "penguins-endpoint",
            "data_capture_percentage": data_capture_percentage,
            "data_capture_destination": data_capture_destination,
            "role": role,
        },
    )
