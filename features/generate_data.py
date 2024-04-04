import os

import pandas as pd
from datasets import load_dataset

from utils.persistent import save_fine_tuning_data
from utils.preprocess import preprocess
import warnings

from utils.prompt import get_smcdel_prompt, get_direct_prompt

warnings.simplefilter(action='ignore', category=DeprecationWarning)


def generate_test_samples(size: int) -> None:
    """
    Generate test samples for the evaluation of the model in TRUE/FALSE balance
    :param size: Number of samples to generate
    :return: None
    """
    assert size > 0, "Size must be greater than 0"
    assert size % 2 == 0, "Size must be even"
    dataset = load_dataset("sileod/mindgames", cache_dir='./cache/data')
    data = pd.DataFrame(dataset['test'])
    data = data.dropna()
    data = preprocess(data)
    data = data.groupby(['target_label']).sample(size // 2, replace=False)
    data.to_csv('./data/test_samples.csv', index=False)
    print(f"Test samples generated successfully in data/test_samples.csv")


def generate_finetuning_samples(size: int, with_sf: bool = True) -> None:
    """
    Generate samples for the LLM fine-tuning task for LLM+Model Checker
    :param size: Number of samples to generate
    :param with_sf: for sf or direct prompt
    :return: None
    """

    # load the dataset
    assert size > 0, "Size must be greater than 0"
    dataset = load_dataset("sileod/mindgames", cache_dir='./cache/data')
    data = pd.DataFrame(dataset['train'])
    data = data.dropna()
    examples = data.sample(data.shape[0] // 10)
    data = data.drop(examples.index)
    data = data.sample(size)

    # preprocess the data
    data = preprocess(data)
    examples = preprocess(examples)

    # prepare fine tune format data
    fine_tune_data = []
    fine_tune_samples = data
    for item in fine_tune_samples.iterrows():
        example = examples.sample(1).iloc[0]
        if with_sf:
            prompt = get_smcdel_prompt(
                example_context=example['context'],
                example_hypothesis=example['hypothesis'],
                example_sf=example['target_sf'],
                problem_context=item[1]['context'],
                problem_hypothesis=item[1]['hypothesis'],
            )
            fine_tune_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": item[1]['target_sf']
                    }
                ]
            }
        else:
            prompt = get_direct_prompt(
                example_context=example['context'],
                example_hypothesis=example['hypothesis'],
                example_answer="TURE" if example['target_label'] == 1 else "FALSE",
                context=item[1]['context'],
                hypothesis=item[1]['hypothesis'],
            )
            fine_tune_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": "TRUE" if item[1]['target_label'] == 1 else "FALSE"
                    }
                ]
            }
        fine_tune_data.append(fine_tune_item)

    save_fine_tuning_data(fine_tune_data, with_sf=with_sf)
