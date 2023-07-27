from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep


def create_sklearn_processor(
    job_name: str,
    framework_version: str,
    instance_type: str,
    instance_count: int,
    role: str,
):
    sklearn_processor = SKLearnProcessor(
        base_job_name=job_name,
        framework_version=framework_version,
        instance_type=instance_type,
        instance_count=instance_count,
        role=role,
    )
    return sklearn_processor


def create_preprocessing_step(
    step_name: str,
    preprocessor,
    cache_config,
    dataset_location,
    preprocessor_destination,
    script_path: str,
):
    # TODO: there is a lot of paths that we need to take care of
    # What outputs are those?
    preprocess_data_step = ProcessingStep(
        name=step_name,
        processor=preprocessor,
        inputs=[
            ProcessingInput(
                source=dataset_location, destination="/opt/ml/processing/input"
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(
                output_name="validation", source="/opt/ml/processing/validation"
            ),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(
                output_name="pipeline",
                source="/opt/ml/processing/pipeline",
                destination=preprocessor_destination,
            ),
            ProcessingOutput(
                output_name="classes",
                source="/opt/ml/processing/classes",
                destination=preprocessor_destination,
            ),
            ProcessingOutput(
                output_name="train-baseline", source="/opt/ml/processing/train-baseline"
            ),
            ProcessingOutput(
                output_name="test-baseline", source="/opt/ml/processing/test-baseline"
            ),
        ],
        code=script_path,
        cache_config=cache_config,
    )
    return preprocess_data_step
