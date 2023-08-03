import json
import time

import boto3

sagemaker = boto3.client("sagemaker")


def lambda_handler(event, context):
    model_package_arn = event["model_package_arn"]
    endpoint_name = event["endpoint_name"]
    data_capture_percentage = event["data_capture_percentage"]
    data_capture_destination = event["data_capture_destination"]
    role = event["role"]

    timestamp = time.strftime("%m%d%H%M%S", time.localtime())
    model_name = f"penguins-model-{timestamp}"
    endpoint_config_name = f"penguins-endpoint-config-{timestamp}"

    sagemaker.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        Containers=[{"ModelPackageName": model_package_arn}],
    )

    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "ModelName": model_name,
                "InstanceType": "ml.t3.medium",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
            }
        ],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": data_capture_percentage,
            "DestinationS3Uri": data_capture_destination,
            "CaptureOptions": [
                {"CaptureMode": "Input"},
                {"CaptureMode": "Output"},
            ],
            "CaptureContentTypeHeader": {
                "JsonContentTypes": ["application/json", "application/octect-stream"]
            },
        },
    )

    sagemaker.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )

    return {"statusCode": 200, "body": json.dumps("Endpoint deployed successfully")}
