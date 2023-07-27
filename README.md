# MLOps with SageMaker ![img](https://raw.githubusercontent.com/awslabs/aws-icons-for-plantuml/main/dist/MachineLearning/SageMakerModel.png)

The ideia of this project is to bring some software engineering best practices to your Machine Learning lifecycle during using SageMaker.

## The code

This code is from Santiago training about SageMaker, I just create a cli to run it step by step, and also test locally before send to the cloud.

The original code was in a notebook, and for that is a little difficult (at least for me) complete seperate the SageMaker learning from the ML side of the stuffs.

## What good stuffs do we have here?

- Pre-commit: validating our code, bring style and linter for the whole project in an automatic way
- Code-completion: by developing this way, we can use github-copilot and all benefits of use an IDE.
- Better structured code: desacopling only ML code from only SageMaker code (WIP).
- Configuration: this way we can centralized all project configuration.
- Patterns: by this we can create different templates for different use cases (computer vision, nlp, traditional ml...) and rappidly shift complete ML Lifecycle Pipelines blazingly fast.

## Requirements

Basically you need to be logged in AWS by using the aws-cli. Another requirement is to pass your role as environment variable.

As this is a specific project that uses penguin dataset, you'll need to have the data into the bucket, and also your role needs access to this data.

- IMG here

To facilitate the access to this env variables, I'm using a .env file that will not be exposed on the repo that has the same format of the file `example.env`.

You need also have a data folder and the data.csv of the project there. As a best practice (files can go wildly big) we'll not versioning the data folder on github.

## How to run the CLI?

After cloned the project, fill up those env variables, you can just go to the command line of the project and run:

```bash
# to run on sagemaker
python src/main.py pipelines run --session session1

# to run locally
python src/main.py pipelines run-local --session session1

```

## Expected results

By running local, you'll see a lot of stuffs going into the data folder. Basically everything that you configured for outputs on your pipeline, you'll be showed there.

By running on sagemaker, you'll see a new job running, and you can investigate what is happening.

## Next Steps

Still working into bring all steps to this repo, currently supporting the step 1.
