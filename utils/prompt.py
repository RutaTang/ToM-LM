import textwrap


def get_smcdel_prompt(example_context: str, example_hypothesis: str, example_sf: str, problem_context: str,
                      problem_hypothesis: str) -> str:
    prompt_template = f"""\
    You should transform the natural language representation to the correct SMCDEL symbolic formulation. You should only output the transformed symbolic formulation as shown in the example.
    ------
    === Example Begin ===
    Natural Language Representation:
    Context:
    {example_context}
    Hypothesis:
    {example_hypothesis}

    Symbolic Formulation:
    {example_sf}
    === Example End ===
    ------
    Natural Language Representation:
    Context:
    {problem_context}
    Hypothesis:
    {problem_hypothesis}

    Symbolic Formulation:
    """
    r = textwrap.dedent(prompt_template)
    return r


def get_direct_prompt(example_context, example_hypothesis, example_answer, context, hypothesis):
    prompt = """\
You will be given a problem with a context and hypothesis. You should judge whether the hypothesis is "TRUE", "FALSE", or "I DON'T KNOW" in the given context. "TRUE" means the hypothesis is true in the context, "FALSE" means the hypothesis is false in the context, and "I DON'T KNOW" means you cannot determine the truth value of the hypothesis in the context. You should provide your answer in the form of "TRUE", "FALSE", or "I DON'T KNOW".
------
Example:

Context:
{example_context}
Hypothesis:
{example_hypothesis}
Answer:
{example_answer}
------
Context: 
{context}
Hypothesis:
{hypothesis}
Answer:
""".format(
        example_context=example_context,
        example_hypothesis=example_hypothesis,
        example_answer=example_answer,
        context=context,
        hypothesis=hypothesis
    )
    return prompt
