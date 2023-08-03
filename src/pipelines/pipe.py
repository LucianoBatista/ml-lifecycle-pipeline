import json

import boto3
from sagemaker.drift_check_baselines import DriftCheckBaselines, MetricsSource
from sagemaker.inputs import TrainingInput
from sagemaker.lambda_helper import Lambda
from sagemaker.model import ModelMetrics, Transformer
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tensorflow import TensorFlow, TensorFlowModel, TensorFlowProcessor
from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import ConditionStep, LambdaStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.steps import (
    CreateModelStep,
    Join,
    JsonGet,
    ProcessingInput,
    ProcessingOutput,
    ProcessingStep,
    TrainingStep,
    TransformStep,
    TuningStep,
)

from config.config import get_settings
from pipelines import pipe_utils


class Pipe:
    def __init__(self):
        # settings from env variables
        self.role = get_settings().execution_role
        self.use_tunning_step = get_settings().use_tuning_step
        self.s3_location = get_settings().s3_location
        self.data_quality_location = get_settings().data_quality_location
        self.model_package_group_name = get_settings().model_package_group_name
        self.preprocessor_script_path = get_settings().preprocessor_script_path
        self.train_script_path = get_settings().train_script_path
        self.evaluate_script_path = get_settings().evaluate_script_path
        self.lambda_script_path = get_settings().lambda_script_path
        self.endpoint_code_path = get_settings().endpoint_code_path
        # base config used across all steps
        self.dataset_location = pipe_utils.get_dataset_location_param()
        self.pipeline_definition_config = pipe_utils.get_pipeline_definition()
        self.cache_config = pipe_utils.get_cache_config()
        self.sagemaker_session = PipelineSession()
        # sagemaker params
        self.param_data_capture_destination = self._get_data_capture_destination_param()
        self.param_data_capture_percentage = self._get_data_capture_percentage_param()
        self.param_accuracy_threshold = self._get_accuracy_threshold_param()
        self.property_evaluation_report = self._get_property_evaluation_report()

    def _get_accuracy_threshold_param(self):
        return ParameterFloat(name="accuracy_threshold", default_value=0.70)

    def _get_data_capture_destination_param(self):
        return ParameterString(
            name="data_capture_destination",
            default_value=self.s3_location + "/monitoring/data-capture",
        )

    def _get_data_capture_percentage_param(self):
        return ParameterInteger(
            name="data_capture_percentage",
            default_value=100,
        )

    def _get_property_evaluation_report(self):
        return PropertyFile(
            name="evaluation-report",
            output_name="evaluation",
            path="evaluation.json",
        )

    def _get_estimator(self) -> TensorFlow:
        """used just on training and tunning steps"""
        estimator = TensorFlow(
            entry_point=self.train_script_path,
            hyperparameters={
                "epochs": 50,
                "batch_size": 32,
            },
            framework_version="2.11",
            instance_type="ml.m5.large",
            py_version="py39",
            instance_count=1,
            script_mode=True,
            # The default profiler rule includes a timestamp which will change each time
            # the pipeline is upserted, causing cache misses. Since we don't need
            # profiling, we can disable it to take advantage of caching.
            disable_profiler=True,
            role=self.role,
            sagemaker_session=self.sagemaker_session,
        )
        return estimator

    def _get_tuner(self, max_jobs: int, max_parallel_jobs: int) -> HyperparameterTuner:
        # configuration of the tuning step
        objective_metric_name = "val_accuracy"
        objective_type = "Maximize"
        metric_definitions = [
            {"Name": objective_metric_name, "Regex": "val_accuracy: ([0-9\\.]+)"}
        ]
        hyperparameter_ranges = {
            "epochs": IntegerParameter(10, 50),
        }
        estimator = self._get_estimator()

        # creation of the tuner class
        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name,
            hyperparameter_ranges,
            metric_definitions,
            objective_type=objective_type,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
        )
        return tuner

    def _create_lambda_role(self, role_name):
        iam_client = boto3.client("iam")
        try:
            response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": "lambda.amazonaws.com"},
                                "Action": "sts:AssumeRole",
                            }
                        ],
                    }
                ),
                Description="Lambda Pipeline Role",
            )

            role_arn = response["Role"]["Arn"]

            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            )

            iam_client.attach_role_policy(
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
                RoleName=role_name,
            )

            return role_arn

        except iam_client.exceptions.EntityAlreadyExistsException:
            response = iam_client.get_role(RoleName=role_name)
            return response["Role"]["Arn"]

    def preprocessing(
        self,
        job_name: str,
        framework_version: str,
        instance_type: str,
        instance_count: int,
        step_name: str,
    ):
        sklearn_processor = SKLearnProcessor(
            base_job_name=job_name,
            framework_version=framework_version,
            instance_type=instance_type,
            instance_count=instance_count,
            role=self.role,
        )
        preprocess_data_step = ProcessingStep(
            name=step_name,
            processor=sklearn_processor,
            inputs=[
                ProcessingInput(
                    source=self.dataset_location, destination="/opt/ml/processing/input"
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train", source="/opt/ml/processing/train"
                ),
                ProcessingOutput(
                    output_name="validation", source="/opt/ml/processing/validation"
                ),
                ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
                ProcessingOutput(
                    output_name="pipeline",
                    source="/opt/ml/processing/pipeline",
                ),
                ProcessingOutput(
                    output_name="classes",
                    source="/opt/ml/processing/classes",
                ),
                ProcessingOutput(
                    output_name="train-baseline",
                    source="/opt/ml/processing/train-baseline",
                ),
                ProcessingOutput(
                    output_name="test-baseline",
                    source="/opt/ml/processing/test-baseline",
                ),
            ],
            code=self.preprocessor_script_path,
            cache_config=self.cache_config,
        )

        return preprocess_data_step

    def data_quality(self, preprocess_data_step: ProcessingStep):
        data_quality_baseline_step = QualityCheckStep(
            name="generate-data-quality-baseline",
            check_job_config=CheckJobConfig(
                instance_type="ml.t3.xlarge",
                instance_count=1,
                volume_size_in_gb=20,
                sagemaker_session=self.sagemaker_session,
                role=self.role,
            ),
            quality_check_config=DataQualityCheckConfig(
                # We will use the train dataset we generated during the preprocessing
                # step to generate the data quality baseline.
                baseline_dataset=preprocess_data_step.properties.ProcessingOutputConfig.Outputs[
                    "train-baseline"
                ].S3Output.S3Uri,
                dataset_format=DatasetFormat.json(lines=True),
                output_s3_uri=self.data_quality_location,
            ),
            skip_check=True,
            register_new_baseline=True,
            model_package_group_name=self.model_package_group_name,
            cache_config=self.cache_config,
        )
        return data_quality_baseline_step

    def training(self, preprocess_data_step: ProcessingStep):
        estimator = self._get_estimator()

        train_model_step = TrainingStep(
            name="train-model",
            step_args=estimator.fit(
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
            cache_config=self.cache_config,
        )
        return train_model_step

    def tuning(self, preprocess_data_step: ProcessingStep):
        tuner = self._get_tuner(max_jobs=3, max_parallel_jobs=3)

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
                },
            ),
            cache_config=self.cache_config,
        )
        return tune_model_step

    def evaluation(
        self,
        preprocess_data_step: ProcessingStep,
        train_model_step: TrainingStep,
        tune_model_step: TuningStep,
    ):
        tensorflow_processor = TensorFlowProcessor(
            base_job_name="penguins-evaluation-processor",
            framework_version="2.6",
            py_version="py38",
            instance_type="ml.m5.large",
            instance_count=1,
            role=self.role,
            sagemaker_session=self.sagemaker_session,
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
                                top_k=0,
                                s3_bucket=self.sagemaker_session.default_bucket(),
                            )
                            if self.use_tunning_step
                            else train_model_step.properties.ModelArtifacts.S3ModelArtifacts
                        ),
                        destination="/opt/ml/processing/model",
                    ),
                ],
                outputs=[
                    ProcessingOutput(
                        output_name="evaluation",
                        source="/opt/ml/processing/evaluation",
                        destination=f"{self.s3_location}/evaluation",
                    ),
                ],
                code=self.evaluate_script_path,
            ),
            property_files=[self.property_evaluation_report],
            cache_config=self.cache_config,
        )
        return evaluate_model_step

    def condition(
        self,
        evaluate_model_step: ProcessingStep,
        create_model_step: ModelStep,
        generate_test_predictions_step,
        model_quality_baseline_step,
        register_model_step,
        deploy_step,
    ):
        fail_step = FailStep(
            name="fail",
            error_message=Join(
                on=" ",
                values=[
                    "Execution failed because the model's accuracy was lower than",
                    self.param_accuracy_threshold,
                ],
            ),
        )

        condition_gte = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluate_model_step.name,
                property_file=self.property_evaluation_report,
                json_path="metrics.accuracy.value",
            ),
            right=self.param_accuracy_threshold,
        )
        condition_step = ConditionStep(
            name="check-model-accuracy",
            conditions=[condition_gte],
            if_steps=[
                create_model_step,
                generate_test_predictions_step,
                model_quality_baseline_step,
                register_model_step,
                deploy_step,
            ],
            else_steps=[fail_step],
        )
        return condition_step

    def _model(
        self,
        preprocess_data_step,
        tune_model_step: TuningStep,
        train_model_step: TrainingStep,
    ):
        model = TensorFlowModel(
            name="penguins",
            model_data=(
                tune_model_step.get_top_model_s3_uri(
                    top_k=0, s3_bucket=self.sagemaker_session.default_bucket()
                )
                if self.use_tunning_step
                else train_model_step.properties.ModelArtifacts.S3ModelArtifacts
            ),
            entry_point="inference.py",
            source_dir=self.endpoint_code_path,
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
            sagemaker_session=self.sagemaker_session,
            role=self.role,
        )
        return model

    def create_model_step(
        self,
        preprocess_data_step,
        tune_model_step: TuningStep,
        train_model_step: TrainingStep,
    ):
        model = self._model(preprocess_data_step, tune_model_step, train_model_step)

        create_model_step = ModelStep(
            name="create",
            display_name="create-model",
            step_args=model.create(instance_type="ml.m5.large"),
        )
        return create_model_step

    def test_predictions(self, preprocess_data_step: ProcessingStep, create_model_step):
        transformer = Transformer(
            model_name=create_model_step.properties.ModelName,
            base_transform_job_name="transform",
            instance_type="ml.c5.xlarge",
            instance_count=1,
            accept="application/json",
            strategy="SingleRecord",
            assemble_with="Line",
            output_path=f"{self.s3_location}/transform",
            sagemaker_session=self.sagemaker_session,
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
                output_filter="$.SageMakerOutput['prediction','groundtruth']",
            ),
            cache_config=self.cache_config,
        )
        return generate_test_predictions_step

    def register(
        self,
        preprocess_data_step,
        data_quality_baseline_step,
        model_quality_baseline_step,
        train_model_step: TrainingStep,
        tune_model_step: TuningStep,
    ):
        model = self._model(preprocess_data_step, tune_model_step, train_model_step)

        model_metrics = ModelMetrics(
            model_data_statistics=MetricsSource(
                s3_uri=data_quality_baseline_step.properties.CalculatedBaselineStatistics,
                content_type="application/json",
            ),
            model_data_constraints=MetricsSource(
                s3_uri=data_quality_baseline_step.properties.CalculatedBaselineConstraints,
                content_type="application/json",
            ),
            model_statistics=MetricsSource(
                s3_uri=model_quality_baseline_step.properties.CalculatedBaselineStatistics,
                content_type="application/json",
            ),
            model_constraints=MetricsSource(
                s3_uri=model_quality_baseline_step.properties.CalculatedBaselineConstraints,
                content_type="application/json",
            ),
        )

        drift_check_baselines = DriftCheckBaselines(
            model_data_statistics=MetricsSource(
                s3_uri=data_quality_baseline_step.properties.BaselineUsedForDriftCheckStatistics,
                content_type="application/json",
            ),
            model_data_constraints=MetricsSource(
                s3_uri=data_quality_baseline_step.properties.BaselineUsedForDriftCheckConstraints,
                content_type="application/json",
            ),
            model_statistics=MetricsSource(
                s3_uri=model_quality_baseline_step.properties.BaselineUsedForDriftCheckStatistics,
                content_type="application/json",
            ),
            model_constraints=MetricsSource(
                s3_uri=model_quality_baseline_step.properties.BaselineUsedForDriftCheckConstraints,
                content_type="application/json",
            ),
        )
        register_model_step = ModelStep(
            name="register",
            display_name="register-model",
            step_args=model.register(
                model_package_group_name=self.model_package_group_name,
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

    def model_quality(self, generate_test_predictions_step: TransformStep):
        model_quality_location = f"{self.s3_location}/monitoring/model-quality"

        model_quality_baseline_step = QualityCheckStep(
            name="generate-model-quality-baseline",
            check_job_config=CheckJobConfig(
                instance_type="ml.t3.xlarge",
                instance_count=1,
                volume_size_in_gb=20,
                sagemaker_session=self.sagemaker_session,
                role=self.role,
            ),
            quality_check_config=ModelQualityCheckConfig(
                # We are going to use the output of the Transform Step to generate
                # the model quality baseline.
                baseline_dataset=generate_test_predictions_step.properties.TransformOutput.S3OutputPath,
                dataset_format=DatasetFormat.json(lines=True),
                # We need to specify the problem type and the fields where the prediction
                # and groundtruth are so the process knows how to interpret the results.
                problem_type="MulticlassClassification",
                inference_attribute="prediction",
                ground_truth_attribute="groundtruth",
                output_s3_uri=model_quality_location,
            ),
            skip_check=True,
            register_new_baseline=True,
            model_package_group_name=self.model_package_group_name,
            cache_config=self.cache_config,
        )
        return model_quality_baseline_step

    def deploy(self, register_model_step: ModelStep):
        lambda_role = self._create_lambda_role("lambda-pipeline-role")

        print(lambda_role)
        deploy_fn = Lambda(
            function_name="deploy_fn",
            execution_role_arn=lambda_role,
            script=self.lambda_script_path,
            handler="lambda.lambda_handler",
            timeout=600,
            session=self.sagemaker_session,
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
                "data_capture_percentage": self.param_data_capture_percentage,
                "data_capture_destination": self.param_data_capture_destination,
                "role": self.role,
            },
        )
        return deploy_step
