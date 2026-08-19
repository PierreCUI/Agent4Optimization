from langchain_openai import ChatOpenAI

from .API import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CodeAgent, CodeWriter


class DeepSeekCodeWriter(CodeWriter):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        model_kwargs = {
            "model": self.model_name,
            "api_key": DEEPSEEK_API_KEY,
            "base_url": DEEPSEEK_BASE_URL,
            "temperature": 0.0,
            "seed": 42,
        }
        self._model = ChatOpenAI(**model_kwargs, extra_body={"enable_thinking": True})


class DeepSeekCodeAgentV4Pro(CodeAgent):
    MODEL_NAME = "deepseek-v4-pro"

    def __init__(self):
        super().__init__()
        self.model_name = DeepSeekCodeAgentV4Pro.MODEL_NAME
        self.writer = DeepSeekCodeWriter(self.model_name)


class DeepSeekCodeAgentV4Flash(CodeAgent):
    MODEL_NAME = "deepseek-v4-flash"

    def __init__(self):
        super().__init__()
        self.model_name = DeepSeekCodeAgentV4Flash.MODEL_NAME
        self.writer = DeepSeekCodeWriter(self.model_name)
