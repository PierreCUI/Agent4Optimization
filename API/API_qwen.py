from langchain_openai import ChatOpenAI

from .API import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, CodeAgent, CodeWriter


class QwenCodeWriter(CodeWriter):
    def __init__(self, model_name):
        if not DASHSCOPE_API_KEY:
            # Fail fast with the provider's name instead of letting the
            # OpenAI SDK resolve OPENAI_API_KEY and blame the wrong one.
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not set — export it before building the Qwen agent."
            )
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


class QwenCodeAgent38Max(CodeAgent):
    MODEL_NAME = "qwen3.8-max"

    def __init__(self):
        super().__init__()
        self.model_name = QwenCodeAgent38Max.MODEL_NAME
        self.writer = QwenCodeWriter(self.model_name)


class QwenCodeAgent38Flash(CodeAgent):
    MODEL_NAME = "qwen3.8-flash"

    def __init__(self):
        super().__init__()
        self.model_name = QwenCodeAgent38Flash.MODEL_NAME
        self.writer = QwenCodeWriter(self.model_name)
