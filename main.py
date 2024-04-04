import os

import click
import numpy as np
import pandas as pd
from datasets import load_dataset

from features.generate_data import generate_test_samples, generate_finetuning_samples
from features.run_model import direct_predict, with_sf_predict
from models.openai_llm import OpenAILLM
from utils.preprocess import preprocess


@click.group()
@click.option("--random-seed", default=1, type=int, help="Random seed", show_default=True)
def app(random_seed: int):
    # initialize random seed
    np.random.seed(random_seed)
    # initialize data directory
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    return


@app.command()
@click.option("--test-samples", help="Generate test samples", is_flag=True)
@click.option("--with-sf-fine-tune", help="Generate fine tune data with sf", is_flag=True)
@click.option("--direct-fine-tune", help="Generate fine tune data for direct", is_flag=True)
@click.option("--size", help="Size of the dataset", type=int, required=True)
def generate_data(test_samples: bool, with_sf_fine_tune: bool, direct_fine_tune: bool, size: int):
    if test_samples:
        generate_test_samples(size=size)
    elif with_sf_fine_tune:
        generate_finetuning_samples(size=size, with_sf=True)
    elif direct_fine_tune:
        generate_finetuning_samples(size=size, with_sf=False)
    else:
        raise ValueError("Please specify the type of data to generate")


@app.command()
@click.option("--model-name", help="Name of the model", type=str, required=True, is_flag=False)
@click.option("--direct", help="Run direct prompt non-fine-tuned model", is_flag=True)
@click.option("--with-sf", help="Run with-sf non-fine-tuned model", is_flag=True)
@click.option("--direct-fine-tune", help="Run direct prompt fine-tuned model", is_flag=True)
@click.option("--with-sf-fine-tune", help="Run with-sf fine-tuned model", is_flag=True)
def run(model_name: str, direct: bool, with_sf: bool, direct_fine_tune: bool, with_sf_fine_tune: bool):
    # load model
    # only gpt-3.5-turbo-0125 and its fine-tuning models are supported
    assert "gpt-3.5-turbo-0125" in model_name
    model = OpenAILLM(model_name=model_name)
    # load test data
    test = pd.read_csv(f"./data/test_samples.csv")
    # sample example from train
    dataset = load_dataset("sileod/mindgames", cache_dir='./cache/data')
    example = pd.DataFrame(dataset['train']).sample(1)
    example = preprocess(example).iloc[0]
    # run
    if direct or direct_fine_tune:
        model.reconfigure(config={"model_name": model_name, "max_tokens": 10})
        direct_predict(model=model, df=test, example=example)
    elif with_sf or with_sf_fine_tune:
        model.reconfigure(config={"model_name": model_name, "max_tokens": 300})
        with_sf_predict(model=model, df=test, example=example)
    else:
        raise ValueError("Please specify the type of model to run")


if __name__ == "__main__":
    app()
