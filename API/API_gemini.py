from langchain_google_genai import ChatGoogleGenerativeAI

from .API import GEMINI_API_KEY, CodeAgent, CodeWriter


class GeminiCodeWriter(CodeWriter):
    def __init__(self, model_name):
        if not GEMINI_API_KEY:
            # Fail fast with the provider's name instead of a confusing
            # key resolution error at request time.
            raise RuntimeError(
                "GEMINI_API_KEY is not set — export it before building the Gemini agent."
            )
        super().__init__()
        self.model_name = model_name
        model_kwargs = {
            "model": self.model_name,
            "google_api_key": GEMINI_API_KEY,
            "temperature": 0.0,
            "seed": 42,
        }
        self._model = ChatGoogleGenerativeAI(**model_kwargs)


class GeminiCodeAgent38Flash(CodeAgent):
    MODEL_NAME = "gemini-3.8-flash"

    def __init__(self):
        super().__init__()
        self.model_name = GeminiCodeAgent38Flash.MODEL_NAME
        self.writer = GeminiCodeWriter(self.model_name)
