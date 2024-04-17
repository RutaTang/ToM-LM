import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.preprocessing import OneHotEncoder


def execution_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the metrics:
    1. Execution rate: whether the predicted symbolic formulation can be executed or not
    2. Execution accuracy: predicted labels match truth labels
    """

    # execution_rate
    er = len(df[df['predicted_label'] != -1]) / len(df)

    # accuracy_rate
    accuracy = accuracy_score(df['target_label'], df['predicted_label'])

    # f1_score
    f1 = f1_score(df['target_label'], df['predicted_label'], average='weighted')

    # auc
    def encode_label(x):
        fail_label = [0, 0]
        true_label = [1, 0]
        false_label = [0, 1]
        if x == 1:
            return true_label
        elif x == 0:
            return false_label
        else:
            return fail_label

    target = df['target_label'].apply(encode_label)
    predicted = df['predicted_label'].apply(encode_label)
    auc = roc_auc_score(target.to_list(), predicted.to_list(), multi_class='ovr', average='weighted')

    m = {
        'execution_accuracy': [accuracy],
        'f1': [f1],
        'auc': [auc],
        'execution_rate': [er],
    }
    mdf = pd.DataFrame(m)
    mdf.fillna(0, inplace=True)
    return mdf


def plot_metrics(df: pd.DataFrame):
    palette = lambda n: sns.color_palette('rocket', n)
    fig, ax = plt.subplots(figsize=(16, 5))
    bars = df.plot(use_index=True, y=list(df.columns), kind='bar', ax=ax, color=palette(len(df.columns)))
    ax.set_ylim(0, 1.2)
    plt.xticks(rotation=0)
    plt.title("Metrics")
    plt.xlabel("Approaches")
    plt.ylabel("Metric Value")
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.2f'),
                      (bar.get_x() + bar.get_width() / 2,
                       bar.get_height()), ha='center', va='center',
                      size=10, xytext=(0, 8),
                      textcoords='offset points')
    plt.show()


def results_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the results distribution
    :param df:
    :return:
    """

    overall = df['predicted_label'].value_counts(normalize=True).reindex([1, 0, -1]).fillna(0)

    return overall


def plot_results_distribution(df: pd.DataFrame):
    # rename columns
    df.rename_axis("Output", axis=1, inplace=True)
    df = df.rename(columns={1: "TRUE", 0: "FALSE", -1: "I DON'T KNOW"})
    # change to percentage
    df = df * 100
    # plot
    palette = lambda n: sns.color_palette('rocket', n)
    fig, ax = plt.subplots(figsize=(16, 5))
    bars = df.plot(use_index=True, y=list(df.columns), kind='bar', ax=ax, color=palette(len(df.columns)))
    ax.set_ylim(0, 100)
    plt.xticks(rotation=0)
    plt.title("Output Distribution")
    plt.xlabel("Approaches")
    plt.ylabel("Percentage")
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.2f'),
                      (bar.get_x() + bar.get_width() / 2,
                       bar.get_height()), ha='center', va='center',
                      size=10, xytext=(0, 8),
                      textcoords='offset points')
    plt.show()
