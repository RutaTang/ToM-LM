import os

import pandas

RESULT_DIR_NAME = "results"
NL2SF_FILE_NAME = "nl2sf.pkl"
SMCDELSF_FILE_NAME = "smcdel_sf.pkl"


def get_result_dir_path(create: bool = True) -> str:
    path = os.path.join(os.getcwd(), RESULT_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def save_nl2sf(nl2sf_df: pandas.DataFrame):
    path = os.path.join(get_result_dir_path(), NL2SF_FILE_NAME)
    nl2sf_df.to_pickle(path)


def load_nl2sf() -> pandas.DataFrame:
    path = os.path.join(get_result_dir_path(), NL2SF_FILE_NAME)
    return pandas.read_pickle(path)


def save_smcdel_sf(df: pandas.DataFrame):
    path = os.path.join(get_result_dir_path(), SMCDELSF_FILE_NAME)
    df.to_pickle(path)


def load_smcdel_sf() -> pandas.DataFrame:
    path = os.path.join(get_result_dir_path(), SMCDELSF_FILE_NAME)
    return pandas.read_pickle(path)
