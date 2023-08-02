# ![img](https://raw.githubusercontent.com/awslabs/aws-icons-for-plantuml/main/dist/MachineLearning/SageMakerModel.png) MLOps with SageMaker

The idea of this repository is to bring best practices from software engineering to your Machine Learning lifecycle process, during the use of **Amazon SageMaker** SDK.

Notebooks are hard to maintain, difficult to diff, and sometimes too large to fit into the GitHub repository. So, a better way to keep track of the code is to break the logic into modules that make sense together, like:

- _Pre-Processing_
- _Feature Engineering_
- _Training_
- _Evaluation_
- _and more..._

All that said, let's understand how you can reproduce this project.

## The code :technologist:

This code is from **ML School - SageMaker Training by Santiago**. I'm just creating a CLI to run it step by step and also using a local machine to test those workflows before sending it to the cloud.

The original code was made in a notebook, and for me, this way was not too intuitive to completely separate the SageMaker concepts from the ML stuff.

## What good stuff do we have here? :tada:

- **Pre-commit**: every commit will trigger some validation on our code and bring style and linter for the whole project in an automatic way.
- **Code-completion**: by developing with the flexibility to code in an IDE, we can use GitHub Copilot and all benefits of code-completion.
- **Better structured code**: a code separate by modules is a lot easier to understand and maintain.
- **Configuration**: we're centralizing all project configurations in one specific module.
- **Patterns**: possibility to create different templates for different use cases (computer vision, NLP, traditional ML...) and rapidly shift from development to production.

## Requirements :wrench:

Basically, you need to be logged into your AWS account using the `aws-cli`.

As this project uses the penguin dataset, you'll need to have the data in the S3 bucket that you configured, and also your `role` will need to have access to this data.

For those who take the training, after running the **setup notebook**, you'll have all you need, except the `aws-cli`.

Using this [medium article](https://medium.com/@harrietty/setting-up-your-aws-account-the-right-way-dfa9a6b5cfbb), you can correctly set up your account and `aws-cli`.

### Environment Variables :closed_lock_with_key:

It is very normal that a lot of projects will need some sensitive information to work properly, like database passwords or API keys for proprietary APIs... One way to manage this is using **environment variables**.

Python has a nice package to help us load specific environment variables, reading those from a file. The package is `python-dotenv`, and you can find it [here](https://pypi.org/project/python-dotenv/).

To indicate which env variables I'm using here, I created a file called `.example.env`, and the only thing you need to do is create a `.env` file and put your information there.

> The project will not upload those values to the repo because the `.env` file was added to the `.gitignore`.

### The data :chart_with_upwards_trend:

You'll need to create a data folder, right after you clone the project, and then put the `data.csv` there. As a best practice (files can go wildly big), we'll not be versioning the data folder on GitHub.

You can go fancy and use [DVC](https://dvc.org) for data versioning, but I'll keep it simple for now.

## Show me how to run it! :fire:

After cloning the project, fill up the env variables and put the data in the correct place, you just need to go to the terminal of the project and run:

```bash
# to run on SageMaker
pipenv install
pipenv shell
python src/main.py pipelines run --session session1

# to run locally
pipenv install
pipenv shell
python src/main.py pipelines run-local --session session1
```

## Expected results :sparkles:

By running locally, you'll see a lot of stuff going into the data folder. Basically everything that you configured for the outputs on your pipeline.

By running on SageMaker, you'll see a new job on the `processing` option on the AWS SageMaker console, and you can investigate what is happening.

## Next Steps :memo:

- [x] Sagemaker code on a OOP-like structure
- [ ] Mapping scripts with the real code for training and others
- [ ] Adjusting resources for the instances

For now, this is a work in progress, and I'm still pulling together the other sessions. Above all, this is just part of my personal learning process, and if you want, feel free to fork the repository and make your own changes.
