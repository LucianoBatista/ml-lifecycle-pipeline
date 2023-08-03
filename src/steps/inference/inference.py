import json
import os
from pathlib import Path
from pickle import load

import boto3
import numpy as np
import pandas as pd
import requests

s3 = boto3.resource("s3")
#
# ENDPOINT_CODE_FOLDER = CODE_FOLDER / "endpoint"
# Path(ENDPOINT_CODE_FOLDER).mkdir(parents=True, exist_ok=True)
# sys.path.append(f"./{ENDPOINT_CODE_FOLDER}")


def handler(
    data,
    context,
    pipeline_file=Path("/tmp") / "pipeline.pkl",
    classes_file=Path("/tmp") / "classes.csv",
):
    """
    This is the entrypoint that will be called by SageMaker when the endpoint
    receives a request. You can see more information at
    https://github.com/aws/sagemaker-tensorflow-serving-container.
    """
    print("Handling endpoint request")

    data = _process_input(data, context, pipeline_file)
    output = _predict(data, context)
    return _process_output(output, context, classes_file)


def _process_input(data, context, pipeline_file):
    print("Processing input data...")

    if context is None:
        # The context will be None when we are testing the code
        # directly from a notebook. In that case, we can use the
        # data directly.
        endpoint_input = data
    elif context.request_content_type in (
        "application/json",
        "application/octet-stream",
    ):
        # When the endpoint is running, we will receive a context
        # object. We need to parse the input and turn it into
        # JSON in that case.
        endpoint_input = json.loads(data.read().decode("utf-8"))

        if endpoint_input is None:
            raise ValueError("There was an error parsing the input request.")
    else:
        raise ValueError(
            f"Unsupported content type: {context.request_content_type or 'unknown'}"
        )

    pipeline = _get_pipeline(pipeline_file)

    df = pd.json_normalize(endpoint_input)
    result = pipeline.transform(df)

    return result[0].tolist()


def _predict(instance, context):
    print("Sending input data to model to make a prediction...")

    model_input = json.dumps({"instances": [instance]})

    if context is None:
        # The context will be None when we are testing the code
        # directly from a notebook. In that case, we want to return
        # a fake prediction back.
        result = {"predictions": [[0.2, 0.5, 0.3]]}
    else:
        # When the endpoint is running, we will receive a context
        # object. In that case we need to send the instance to the
        # model to get a prediction back.
        response = requests.post(context.rest_uri, data=model_input)

        if response.status_code != 200:
            raise ValueError(response.content.decode("utf-8"))

        result = json.loads(response.content)

    print(f"Response: {result}")
    return result


def _process_output(output, context, classes_file):
    print("Processing prediction received from the model...")

    response_content_type = (
        "application/json" if context is None else context.accept_header
    )

    prediction = np.argmax(output["predictions"][0])
    confidence = output["predictions"][0][prediction]

    print(f"Prediction: {prediction}. Confidence: {confidence}")

    result = (
        json.dumps(
            {
                "species": _get_class(prediction, classes_file),
                "prediction": int(prediction),
                "confidence": confidence,
            }
        ),
        response_content_type,
    )

    return result


def _get_pipeline(pipeline_file):
    """
    This function returns the Scikit-Learn pipeline we used to transform the
    dataset.
    """

    _download(pipeline_file, os.environ.get("PIPELINE_S3_LOCATION", None))
    return load(open(pipeline_file, "rb"))


def _get_class(prediction, classes_file):
    """
    This function returns the class name of a given prediction.
    """

    _download(classes_file, os.environ.get("CLASSES_S3_LOCATION", None))

    with open(classes_file) as f:
        file = f.readlines()

    classes = list(map(lambda x: x.replace("'", ""), file[0].split(",")))
    return classes[prediction]


def _download(file, s3_location):
    """
    This function downloads a file from S3 if it doesn't already exist.
    """
    if file.exists():
        return

    s3_parts = s3_location.split("/", 3)
    bucket = s3_parts[2]
    key = s3_parts[3]

    s3.Bucket(bucket).download_file(key, str(file))
