import random
from datetime import datetime

from IPython.display import JSON
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.model import Model
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.s3 import S3Uploader
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import ParameterBoolean
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)

DATA_QUALITY_LOCATION = f"{S3_LOCATION}/monitoring/data-quality"
files = S3Downloader.list(data_capture_destination.default_value)[:3]
if len(files):
    lines = S3Downloader.read_file(files[0])
    print(json.dumps(json.loads(lines.split("\n")[0]), indent=2))


def create_baseline_step():
    data_quality_baseline_step = QualityCheckStep(
        name="generate-data-quality-baseline",
        check_job_config=CheckJobConfig(
            instance_type="ml.t3.xlarge",
            instance_count=1,
            volume_size_in_gb=20,
            sagemaker_session=sagemaker_session,
            role=role,
        ),
        quality_check_config=DataQualityCheckConfig(
            # We will use the train dataset we generated during the preprocessing
            # step to generate the data quality baseline.
            baseline_dataset=preprocess_data_step.properties.ProcessingOutputConfig.Outputs[
                "train-baseline"
            ].S3Output.S3Uri,
            dataset_format=DatasetFormat.json(lines=True),
            output_s3_uri=DATA_QUALITY_LOCATION,
        ),
        skip_check=True,
        register_new_baseline=True,
        model_package_group_name=model_package_group_name,
        cache_config=cache_config,
    )
    return data_quality_baseline_step
