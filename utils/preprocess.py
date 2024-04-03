import copy

import pandas as pd

from utils import join_names, insert_sentence_after_period


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data and only keep the necessary columns
    :param data
    :return:
    """

    data = copy.deepcopy(data)

    # Preprocess context and question
    data['formatted_names'] = data['names'].apply(lambda x: join_names(x))
    data['formatted_names'] = data['formatted_names'].apply(lambda x: f"Their names are {x}.")
    data['context'] = data[['premise', 'formatted_names']].apply(
        lambda x: insert_sentence_after_period(x['premise'], x['formatted_names']), axis=1)

    # Preprocess smcdel
    data['target_sf'] = data['smcdel_problem']

    # label
    data['target_label'] = data['label'].apply(lambda x: 1 if x == 'entailment' else 0)

    # setup
    data['setup'] = data['setup'].astype('category')

    # Drop unnecessary columns
    keep_columns = ['setup', 'context', 'hypothesis', 'target_sf', 'target_label', 'n_announcements', 'n_agents',
                    'hypothesis_depth']
    data = data[keep_columns]

    return data
