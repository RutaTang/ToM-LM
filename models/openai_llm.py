import os
import unittest

import dotenv
import openai

if __name__ == "__main__":
    from base_llm import BaseLLM
else:
    from .base_llm import BaseLLM


class OpenAILLM(BaseLLM):

    def __init__(self, model_name: str):
        # load openai key from .env file
        dotenv.load_dotenv(dotenv.find_dotenv())
        # set model name
        self.model_name = model_name
        self.max_tokens = 150

    def reconfigure(self, config: dict):
        self.max_tokens = config.get("max_tokens", 150)
        self.model_name = config.get("model_name", "gpt-3.5-turbo")

    def complete(self, text: str) -> str:
        client = openai.Client(api_key=os.environ["OPENAI_KEY"])
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": text
                }
            ],
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=0
        )
        return response.choices[0].message.content


class TestOpenAILLM(unittest.TestCase):

    def test_complete(self):
        model = OpenAILLM("gpt-3.5-turbo")
        text = "What is the meaning of life?"
        response = model.complete(text)
        print(response)
        self.assertTrue(len(response) > 0)


if __name__ == "__main__":
    unittest.main()
