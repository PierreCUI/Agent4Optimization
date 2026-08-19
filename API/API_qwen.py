from langchain_openai import ChatOpenAI

from .API import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, CodeAgent, CodeWriter


class QwenCodeWriter(CodeWriter):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        model_kwargs = {
            "model": self.model_name,
            "api_key": DASHSCOPE_API_KEY,
            "base_url": DASHSCOPE_BASE_URL,
            "temperature": 0.0,
            "seed": 42,
        }
        self._model = ChatOpenAI(**model_kwargs, extra_body={"enable_thinking": True})


class QwenCodeAgent37Max(CodeAgent):
    MODEL_NAME = "qwen3.7-max"

    def __init__(self):
        super().__init__()
        self.model_name = QwenCodeAgent37Max.MODEL_NAME
        self.writer = QwenCodeWriter(self.model_name)


class QwenCodeAgent37Plus(CodeAgent):
    MODEL_NAME = "qwen3.7-plus"

    def __init__(self):
        super().__init__()
        self.model_name = QwenCodeAgent37Plus.MODEL_NAME
        self.writer = QwenCodeWriter(self.model_name)
