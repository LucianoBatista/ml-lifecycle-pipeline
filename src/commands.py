from typing import Optional

from typer import Typer

from config.config import get_settings
from setup import pipe_1, pipe_2

app = Typer()

SESSION_DICT = {"session1": pipe_1, "session2": pipe_2}


@app.command()
def run(session: Optional[str] = "session1"):
    # env variables
    role = get_settings().execution_role

    # upserting the pipeline
    pipeline = SESSION_DICT[session]
    pipeline_to_run = pipeline(role)
    pipeline_to_run.upsert(role_arn=role)
    pipeline_to_run.start()


@app.command()
def run_local(session: Optional[str] = "session1"):
    BASE_FILEPATH = "data"
    DATA_FILEPATH = "data/data.csv"
    TRAIN_PATH = "data/train"
    VALIDATION_PATH = "data/validation"
    EPOCHS = 10

    if session == "session1":
        from steps.preprocessing import preprocess

        preprocess(base_dir=BASE_FILEPATH, data_filepath=DATA_FILEPATH)

    elif session == "session2":
        from steps.preprocessing import preprocess
        from steps.training import train

        preprocess(base_dir=BASE_FILEPATH, data_filepath=DATA_FILEPATH)
        train(
            base_directory=BASE_FILEPATH,
            train_path=TRAIN_PATH,
            validation_path=VALIDATION_PATH,
            epochs=EPOCHS,
        )
