import unittest

import requests

from models.base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):

    def complete(self, text: str) -> str:
        API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder2-15b"
        headers = {"Authorization": "Bearer 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"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        output = query({
            "inputs": text,
        })

        return output[0]['generated_text']


class TestHuggingFaceLLM(unittest.TestCase):
    def test_complete(self):
        print("Testing HuggingFaceLLM")
        llm = HuggingFaceLLM()
        completion = llm.complete("def add(a, b):")
        print(completion)
        self.assertTrue(len(completion) > 0)


if __name__ == '__main__':
    unittest.main()
