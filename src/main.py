import runner
from typer import Typer


def create_typer_app() -> Typer:
    app = Typer(help="A CLI for running sagemaker pipelines")
    app.add_typer(runner.app, name="pipelines")
    return app


if __name__ == "__main__":
    app = create_typer_app()
    app()
#     run(
#         "session1",
#         "arn:aws:iam::726872986818:role/service-role/SageMaker-MLOpsEngineer",
#     )
