from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.steps import TrainingStep
from sagemaker.inputs import TrainingInput


def create_tensorflow_estimator(role, CODE_FOLDER):
    estimator = TensorFlow(
        entry_point=f"{CODE_FOLDER}/train.py",
        hyperparameters={"epochs": 50, "batch_size": 32},
        framework_version="2.6",
        instance_type="ml.m5.large",
        py_version="py38",
        instance_count=1,
        script_mode=True,
        # The default profiler rule includes a timestamp which will change each time
        # the pipeline is upserted, causing cache misses. Since we don't need
        # profiling, we can disable it to take advantage of caching.
        disable_profiler=True,
        role=role,
    )
    return estimator


def create_training_step(estimator, preprocess_data_step, cache_config):
    train_model_step = TrainingStep(
        name="train-model",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocess_data_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=preprocess_data_step.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )
    return train_model_step
