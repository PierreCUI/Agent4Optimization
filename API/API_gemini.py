from langchain_google_genai import ChatGoogleGenerativeAI

from .API import GEMINI_API_KEY, CodeAgent, CodeWriter


class GeminiCodeWriter(CodeWriter):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        model_kwargs = {
            "model": self.model_name,
            "google_api_key": GEMINI_API_KEY,
            "temperature": 0.0,
            "seed": 42,
        }
        self._model = ChatGoogleGenerativeAI(**model_kwargs)


class GeminiCodeAgent36Flash(CodeAgent):
    MODEL_NAME = "gemini-3.6-flash"

    def __init__(self):
        super().__init__()
        self.model_name = GeminiCodeAgent36Flash.MODEL_NAME
        self.writer = GeminiCodeWriter(self.model_name)


class GeminiCodeAgent35Flash(CodeAgent):
    MODEL_NAME = "gemini-3.5-flash"

    def __init__(self):
        super().__init__()
        self.model_name = GeminiCodeAgent35Flash.MODEL_NAME
        self.writer = GeminiCodeWriter(self.model_name)
