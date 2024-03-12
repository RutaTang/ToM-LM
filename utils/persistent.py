import os
import unittest

import pandas
from typing import TypedDict

RESULT_DIR_NAME = "results"
NL2SF_FILE_NAME = "nl2sf.pkl"
SMCDELSF_FILE_NAME = "smcdel_sf.pkl"


class NameInfo(TypedDict):
    """
    Used for persisting the result
    """
    model_name: str
    sample_size: int
    random_state: int


def get_result_dir_path() -> str:
    path = os.path.join(os.getcwd(), RESULT_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _get_result_file_path(name_info: NameInfo, file_name: str) -> str:
    path = get_result_dir_path()
    name = f"{name_info['model_name']}_{name_info['sample_size']}_{name_info['random_state']}_{file_name}"
    return os.path.join(path, name + file_name)


def save_nl2sf(nl2sf_df: pandas.DataFrame, name_info: NameInfo):
    path = _get_result_file_path(name_info, NL2SF_FILE_NAME)
    nl2sf_df.to_pickle(path)


def load_nl2sf(name_info: NameInfo) -> pandas.DataFrame:
    path = _get_result_file_path(name_info, NL2SF_FILE_NAME)
    return pandas.read_pickle(path)


def save_smcdel_sf(df: pandas.DataFrame, name_info: NameInfo):
    path = _get_result_file_path(name_info, SMCDELSF_FILE_NAME)
    df.to_pickle(path)


def load_smcdel_sf(name_info: NameInfo) -> pandas.DataFrame:
    path = _get_result_file_path(name_info, SMCDELSF_FILE_NAME)
    return pandas.read_pickle(path)


class TestPersistent(unittest.TestCase):

    def test_name_info(self):
        name_info: NameInfo = {
            "model_name": "test_model",
            "sample_size": 100,
            "random_state": 42,
        }


if __name__ == "__main__":
    unittest.main()
