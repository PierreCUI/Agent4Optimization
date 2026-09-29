from langchain_openai import ChatOpenAI

from .API import OPENAI_API_KEY, OPENAI_BASE_URL, CodeAgent, CodeWriter


class ChatGPTCodeWriter(CodeWriter):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        model_kwargs = {
            "model": self.model_name,
            "api_key": OPENAI_API_KEY,
            "base_url": OPENAI_BASE_URL,
            "temperature": 0.0,
            "seed": 42,
        }
        self._model = ChatOpenAI(**model_kwargs)


class ChatGPTCodeAgent6Astra(CodeAgent):
    MODEL_NAME = "gpt-6-astra"

    def __init__(self):
        super().__init__()
        self.model_name = ChatGPTCodeAgent6Astra.MODEL_NAME
        self.writer = ChatGPTCodeWriter(self.model_name)


class ChatGPTCodeAgent6Sol(CodeAgent):
    MODEL_NAME = "gpt-6-sol"

    def __init__(self):
        super().__init__()
        self.model_name = ChatGPTCodeAgent6Sol.MODEL_NAME
        self.writer = ChatGPTCodeWriter(self.model_name)


class ChatGPTCodeAgent6Luna(CodeAgent):
    MODEL_NAME = "gpt-6-luna"

    def __init__(self):
        super().__init__()
        self.model_name = ChatGPTCodeAgent6Luna.MODEL_NAME
        self.writer = ChatGPTCodeWriter(self.model_name)
