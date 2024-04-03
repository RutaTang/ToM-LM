import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score


def execution_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the metrics:
    1. Execution rate: whether the predicted symbolic formulation can be executed or not
    2. Execution accuracy: predicted labels match truth labels
    3. AUC: Area under the ROC curve
    4. F1 score: F1 score
    """

    # execution_rate
    er = df.groupby('setup').apply(
        lambda x: len(x[x['predicted_label'] != -1]) / len(x),
        include_groups=False
    )
    er['overall'] = len(df[df['predicted_label'] != -1]) / len(df)

    # drop rows with predicted_label = -1
    df = df[df['predicted_label'] != -1]

    # accuracy_rate
    accuracy = df.groupby('setup').apply(
        lambda x: accuracy_score(x['target_label'], x['predicted_label']),
        include_groups=False,
    )
    accuracy['overall'] = accuracy_score(df['target_label'], df['predicted_label'])

    # auc
    auc = df.groupby('setup').apply(
        lambda x: roc_auc_score(x['target_label'], x['predicted_label'] ),
        include_groups=False,
    )
    auc['overall'] = roc_auc_score(df['target_label'], df['predicted_label'] )

    # f1
    f1 = df.groupby('setup').apply(
        lambda x: f1_score(x['target_label'], x['predicted_label']),
        include_groups=False,
    )
    f1['overall'] = f1_score(df['target_label'], df['predicted_label'])

    m = {
        'execution_accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'execution_rate': er,
    }
    mdf = pd.DataFrame(m)
    mdf.fillna(0, inplace=True)
    return mdf


def plot_metrics(df: pd.DataFrame):
    palette = lambda n: sns.color_palette('rocket', n)
    df = execution_metrics(df)
    fig, ax = plt.subplots(figsize=(16, 5))
    bars = df.plot(use_index=True, y=list(df.columns), kind='bar', ax=ax, color=palette(len(df.columns)))
    ax.set_ylim(0, 1.2)
    plt.xticks(rotation=0)
    plt.title("Metrics")
    plt.xlabel("Setups")

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

    distribution = df.groupby('setup').apply(
        lambda x: x['predicted_label'].value_counts(normalize=True).reindex([1, 0, -1], fill_value=0),
        include_groups=False
    ).stack()

    overall = df['predicted_label'].value_counts(normalize=True).reindex([1, 0, -1], fill_value=0)

    distribution.loc[('overall', 1)] = overall.loc[1]
    distribution.loc[('overall', 0)] = overall.loc[0]
    distribution.loc[('overall', -1)] = overall.loc[-1]

    distribution.rename('percentage', inplace=True)
    distribution.sort_index(inplace=True, ascending=False)
    distribution.fillna(0, inplace=True)
    return distribution
