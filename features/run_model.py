import pandas as pd

from executors.smcdel import SMCDEL
from models.base_llm import BaseLLM
from utils.persistent import save_direct_result, NameInfo, save_smcdel_result
from utils.prompt import get_direct_prompt, get_smcdel_prompt
from tqdm import tqdm

tqdm.pandas()


def direct_predict(model: BaseLLM, df: pd.DataFrame, example: pd.Series) -> None:
    def predict(row: pd.Series):
        # form prompt
        context = row['context']
        hypothesis = row['hypothesis']
        prompt = get_direct_prompt(
            example_context=example['context'],
            example_hypothesis=example['hypothesis'],
            example_answer="TURE" if example['target_label'] == 1 else "FALSE",
            context=context,
            hypothesis=hypothesis
        )
        # do completion
        answer = model.complete(prompt)
        if answer == "TRUE":
            answer = 1
        elif answer == "FALSE":
            answer = 0
        else:
            answer = -1
        # store predicted symbolic formulation to original df
        row['predicted_label'] = answer
        return row

    df = df.copy()
    df['predicted_label'] = -1
    results = df.progress_apply(predict, axis=1)
    save_direct_result(results, name_info=NameInfo(
        model_name=model.get_name(),
        sample_size=len(df),
    ))


def with_sf_predict(model: BaseLLM, df: pd.DataFrame, example: pd.Series) -> None:
    def predict(row):
        # form prompt
        context = row['context']
        hypothesis = row['hypothesis']
        prompt = get_smcdel_prompt(
            example_context=example['context'],
            example_hypothesis=example['hypothesis'],
            example_sf=example['target_sf'],
            problem_context=context,
            problem_hypothesis=hypothesis
        )
        # do completion
        sf = model.complete(prompt)
        # use smcdel to execute symbolic formulation
        try:
            result = SMCDEL(text=sf)
            result = 1 if result else 0
        except ValueError as e:
            result = -1
        row['predicted_label'] = result
        return row

    df = df.copy()
    df['predicted_label'] = -1
    results = df.progress_apply(predict, axis=1)
    save_smcdel_result(results, name_info=NameInfo(
        model_name=model.get_name(),
        sample_size=len(df),
    ))
