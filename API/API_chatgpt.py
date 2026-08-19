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


class ChatGPTCodeAgent56Sol(CodeAgent):
    MODEL_NAME = "gpt-5.6-sol"

    def __init__(self):
        super().__init__()
        self.model_name = ChatGPTCodeAgent56Sol.MODEL_NAME
        self.writer = ChatGPTCodeWriter(self.model_name)


class ChatGPTCodeAgent56Terra(CodeAgent):
    MODEL_NAME = "gpt-5.6-terra"

    def __init__(self):
        super().__init__()
        self.model_name = ChatGPTCodeAgent56Terra.MODEL_NAME
        self.writer = ChatGPTCodeWriter(self.model_name)


class ChatGPTCodeAgent56Luna(CodeAgent):
    MODEL_NAME = "gpt-5.6-luna"

    def __init__(self):
        super().__init__()
        self.model_name = ChatGPTCodeAgent56Luna.MODEL_NAME
        self.writer = ChatGPTCodeWriter(self.model_name)
