import json
import os
import unittest

import pandas
from typing import TypedDict

RESULT_DIR_NAME = "results"
DATA_DIR_NAME = "data"
NL2SF_FILE_NAME = "nl2sf"
SMCDEL_RESULT_FILE_NAME = "smcdel_result"
DIRECT_RESULT_FILE_NAME = "direct_result"
FINE_TUNING_DATA_FILE_NAME = "fine_tuning_data"


class NameInfo(TypedDict):
    """
    Used for persisting the result
    """
    model_name: str
    sample_size: int


def get_result_dir_path() -> str:
    path = os.path.join(os.getcwd(), RESULT_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def get_data_dir_path() -> str:
    path = os.path.join(os.getcwd(), DATA_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _get_result_file_path(name_info: NameInfo, file_name: str) -> str:
    path = get_result_dir_path()
    name = f"{name_info['model_name']}_{name_info['sample_size']}_{file_name}.pkl"
    return os.path.join(path, name)


def save_nl2sf(nl2sf_df: pandas.DataFrame, name_info: NameInfo):
    path = _get_result_file_path(name_info, NL2SF_FILE_NAME)
    nl2sf_df.to_pickle(path)


def load_nl2sf(name_info: NameInfo) -> pandas.DataFrame:
    path = _get_result_file_path(name_info, NL2SF_FILE_NAME)
    return pandas.read_pickle(path)


def save_smcdel_result(df: pandas.DataFrame, name_info: NameInfo):
    path = _get_result_file_path(name_info, SMCDEL_RESULT_FILE_NAME)
    df.to_pickle(path)


def load_smcdel_result(name_info: NameInfo) -> pandas.DataFrame:
    path = _get_result_file_path(name_info, SMCDEL_RESULT_FILE_NAME)
    return pandas.read_pickle(path)


def save_direct_result(df: pandas.DataFrame, name_info: NameInfo):
    path = _get_result_file_path(name_info, DIRECT_RESULT_FILE_NAME)
    df.to_pickle(path)


def load_direct_result(name_info: NameInfo) -> pandas.DataFrame:
    path = _get_result_file_path(name_info, DIRECT_RESULT_FILE_NAME)
    return pandas.read_pickle(path)


def save_fine_tuning_data(data, with_sf: bool):
    path = get_data_dir_path()
    with_sf = "with_sf" if with_sf else "without_sf"
    path = os.path.join(path, f"{FINE_TUNING_DATA_FILE_NAME}_{with_sf}.jsonl")
    # to jsonl
    with open(path, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')
    print(f"Fine tuning data saved to {path}")


class TestPersistent(unittest.TestCase):

    def test_name_info(self):
        name_info: NameInfo = {
            "model_name": "test_model",
            "sample_size": 100,
        }
        path = _get_result_file_path(name_info, NL2SF_FILE_NAME)
        print(path)


if __name__ == "__main__":
    unittest.main()
