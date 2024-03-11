from typing import List


def insert_sentence_after_period(paragraph: str, new_sentence: str) -> str:
    """
    Insert a new sentence after the first period in the paragraph
    :param paragraph
    :param new_sentence
    :return
    """
    period_index = paragraph.find('.')

    if period_index != -1:
        modified_paragraph = paragraph[:period_index + 1] + ' ' + new_sentence + paragraph[period_index + 1:]
        return modified_paragraph
    else:
        raise ValueError("Period not found in the paragraph")


def join_names(names: List[str]) -> str:
    """
    Join names with a comma and an "and" before the last name
    :param names
    :return:
    """
    if len(names) == 0:
        return ""
    elif len(names) == 1:
        return names[0]
    else:
        return ", ".join(names[:-1]) + ", and " + names[-1]


join_names(["John", "Jane", "Jack"])
