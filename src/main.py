from typer import Typer

import commands


def create_typer_app() -> Typer:
    app = Typer(help="A CLI for running sagemaker pipelines")
    app.add_typer(commands.app, name="pipelines")
    return app


if __name__ == "__main__":
    app = create_typer_app()
    app()
