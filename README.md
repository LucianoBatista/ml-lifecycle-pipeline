# ![img](https://raw.githubusercontent.com/awslabs/aws-icons-for-plantuml/main/dist/MachineLearning/SageMakerModel.png) MLOps with SageMaker

The ideia of this repository is to bring best practices from software engineering to your Machine Learning lifecycle, during the use of **Amazon SageMaker** sdk.

Notebooks are hard to maintain, difficult to diff and sometimes too large to fit into the github repository. So, a better way to keep track of the code is to break the logic by modules that make sense together, like:

- _Pre-Processing_
- _Feature Engineering_
- _Training_
- _Evaluation_
- _and more..._

All that said, let's understand how you can reproduce this project.

## The code

This code is from **ML School - SageMaker Training by Santiago**, I just create a cli to run it step by step, and also using a locally machine to test those workflows before send it to the cloud.

The original code was made in a notebook, and for me, this way was not too intuitive to complete seperate the SageMaker concepts from the ML stuffs.

## What good stuffs do we have here?

- **Pre-commit**: every commit will trigger some validation on our code and bring style and linter for the whole project in an automatic way.
- **Code-completion**: by developing with the flexibility to code in a IDE we can use github-copilot and all benefits of code-completion.
- **Better structured code**: a code seperate by modules is a lot easier to understand and maintain.
- **Configuration**: we're centralizing all project configuration in one specific module.
- **Patterns**: possibility to create different templates for different use cases (computer vision, nlp, traditional ml...) and rappidly shift from development to production.

## Requirements

Basically you need to be logged in your AWS account by using the `aws-cli`.

As this project uses penguin dataset, you'll need to have the data into the s3 bucket that you configured, and also your role will need access to this data.

For those who take the training, after running the setup notebook, you'll have all you need, except the `aws-cli`

Using this [medium article](https://medium.com/@harrietty/setting-up-your-aws-account-the-right-way-dfa9a6b5cfbb) you can correctly setup your account and `aws-cli`

### Environment Variables

Is very normal, that a lot of projects will need some sensible information to work properly, like database passwords, or api-keys for proprietary APIs... One way to manage this is using **environment variables**.

Python has a nice package to help us load specific environment variables, reading those from a file. Tha package is `python-dotenv` and you can find [here](https://pypi.org/project/python-dotenv/).

To indicate which env variables I'm using here, I created a file colled `.example.env`, and the only thing you need to do is create a `.env` file and put your information there.

> The project will not upload those values to the repo, because the `.env` file was added to the `.gitignore`.

### The data

You'll need to create data folder, right after you clone the project, and than put the `data.csv` there. As a best practice (files can go wildly big) and we'll not be versioning the data folder on github.

You can go fancy and use [DVC](https://dvc.org) for data versioning, but I'll keep it simple for now.

## Show me how to run it!

After cloned the project, fill up the env variables and put the data on the correct place, you just need to go to the terminal of the project and run:

```bash
# to run on sagemaker
pipenv install
pipenv shell
python src/main.py pipelines run --session session1

# to run locally
pipenv install
pipenv shell
python src/main.py pipelines run-local --session session1

```

## Expected results

By running local, you'll see a lot of stuffs going into the data folder. Basically everything that you configured for the outputs on your pipeline.

By running on sagemaker, you'll see a new job on the `processing` option on the aws sagemaker console, and you can investigate what is happening.

## Next Steps

For now, this is a working in progress, and I'm still pulling together the others sessions. Above all, this is just part of my personal learning process, and if you want feel free to fork the repository and do your own changes.
