from sagemaker.tensorflow import TensorFlowModel
from sagemaker.workflow.model_step import ModelStep

USE_TUNING_STEP = False


def create_registration_step(
    pipeline_session,
    role,
    train_model_step,
    tune_model_step,
    model_package_group_name,
    model_metrics,
    sagemaker_session,
):
    model = TensorFlowModel(
        model_data=(
            tune_model_step.get_top_model_s3_uri(
                top_k=0, s3_bucket=sagemaker_session.default_bucket()
            )
            if USE_TUNING_STEP
            else train_model_step.properties.ModelArtifacts.S3ModelArtifacts
        ),
        framework_version="2.6",
        sagemaker_session=pipeline_session,
        role=role,
    )

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


def create_another_registration():
    register_model_step = ModelStep(
        name="register",
        display_name="register-model",
        step_args=model.register(
            model_package_group_name=model_package_group_name,
            model_metrics=model_metrics,
            drift_check_baselines=drift_check_baselines,
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
    return register_model_step
