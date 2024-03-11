import textwrap


def get_prompt(example_context: str, example_hypothesis: str, example_sf: str, problem_context: str,
               problem_hypothesis: str) -> str:
    prompt_template = f"""\
    You should transform the natural language representation to the correct SMCDEL symbolic formulation. You should only output the transformed symbolic formulation as shown in the example.
    ------
    === Example Begin ===
    Natural Language Representation:
    * Problem Context: {example_context}
    * Hypothesis for validation: {example_hypothesis}

    SMCDEL Symbolic Formulation:
    {example_sf}
    === Example End ===
    ------
    Natural Language Representation:
    * Problem Context: {problem_context}
    * Hypothesis for validation: {problem_hypothesis}

    SMCDEL Symbolic Formulation:
    """
    r = textwrap.dedent(prompt_template)
    return r
