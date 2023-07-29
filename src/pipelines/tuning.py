from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker.workflow.steps import TuningStep
from sagemaker.inputs import TrainingInput


def create_tuner(estimator):
    objective_metric_name = "val_accuracy"
    objective_type = "Maximize"
    metric_definitions = [
        {"Name": objective_metric_name, "Regex": "val_accuracy: ([0-9\\.]+)"}
    ]

    hyperparameter_ranges = {
        "epochs": IntegerParameter(10, 50),
    }

    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions,
        objective_type=objective_type,
        max_jobs=3,
        max_parallel_jobs=3,
    )
    return tuner


def create_tuner_step(preprocess_data_step, tuner, cache_config):
    tune_model_step = TuningStep(
        name="tune-model",
        step_args=tuner.fit(
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
            }
        ),
        cache_config=cache_config,
    )
    return tune_model_step
