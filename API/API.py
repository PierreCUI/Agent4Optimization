import json
import os
import time
import warnings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.openai.com/v1",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class CodeWriter:
    def __init__(self):
        self.model_name = None
        self._model = None
        self._prompt = None
        self._parser = StrOutputParser()
        self._chain = None

    def initialization(self, prompt_list=None, partial_dict=None):
        self._prompt = ChatPromptTemplate.from_messages(prompt_list)
        if partial_dict:
            self._prompt = self._prompt.partial(**partial_dict)
        self._chain = self._prompt | self._model

    def invoke(self, function_dict):
        response = self._chain.invoke(function_dict)
        try:
            if hasattr(response, "usage_metadata"):
                print(
                    f"Usage: "
                    f"Input {response.usage_metadata['input_tokens']} "
                    f"Output {response.usage_metadata['output_tokens']}."
                )
            elif hasattr(response, "response_metadata"):
                print(
                    f"Usage: "
                    f"Input {response.response_metadata['input_tokens']} "
                    f"Output {response.response_metadata['output_tokens']}."
                )
            else:
                warnings.warn(
                    f"Response object has no usage_metadata or response_metadata. "
                    f"Token usage unavailable (model: {self.model_name})."
                )
        except (KeyError, TypeError) as e:
            warnings.warn(f"Failed to parse token usage from response metadata: {e}")
        return self._parser.invoke(response)


class CodeAgent:
    MODEL_NAME = None

    def __init__(self):
        self.model_name = None
        self.writer = None
        self.prompt_list = []
        self.history = []
        self.history_file = f"./history/chat_{time.time()}.json"
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

    def invoke(self, function_dict, partial_dict=None):
        self.writer.initialization(self.prompt_list, partial_dict=partial_dict)
        full_text = self.writer.invoke(function_dict)
        self.history.append({"input": self.prompt_list, "output": full_text})
        self.save_history()
        return full_text

    def set_prompt_list(self, prompt_list):
        self.prompt_list = prompt_list

    def save_history(self):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)
