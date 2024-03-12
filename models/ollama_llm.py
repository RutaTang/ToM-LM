import unittest

import ollama

if __name__ == "__main__":
    from base_llm import BaseLLM
else:
    from .base_llm import BaseLLM


class OllamaLLM(BaseLLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    def complete(self, text: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": text
            }],
            options={
                'temperature': 0.7,
            },
            stream=False
        )
        response = response['message']['content']
        return response


class TestOllamaLLM(unittest.TestCase):
    def test_complete(self):
        print("Testing OllamaLLM")
        llm = OllamaLLM('gemma')
        completion = llm.complete("Hello!")
        print(completion)
        self.assertTrue(len(completion) > 0)


if __name__ == '__main__':
    unittest.main()
