from langchain_openai import ChatOpenAI

from .API import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CodeAgent, CodeWriter


class DeepSeekCodeWriter(CodeWriter):
    def __init__(self, model_name):
        if not DEEPSEEK_API_KEY:
            # Without this guard the OpenAI SDK silently resolves
            # OPENAI_API_KEY instead and the run dies later with a 401
            # that looks like a DeepSeek outage.
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set — export it before building the DeepSeek agent."
            )
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
    # v4.1 flash interface
    MODEL_NAME = "deepseek-flash"

    def __init__(self):
        super().__init__()
        self.model_name = DeepSeekCodeAgentV4Flash.MODEL_NAME
        self.writer = DeepSeekCodeWriter(self.model_name)


class DeepSeekCodeAgentV4FlashVision(CodeAgent):
    # Vision-capable variant of the flash model: accepts image inputs
    # via CodeWriter.invoke_multimodal (used by the reflection loop with
    # the per-snapshot design images).
    MODEL_NAME = "deepseek-v4-flash-vision-exp"

    def __init__(self):
        super().__init__()
        self.model_name = DeepSeekCodeAgentV4FlashVision.MODEL_NAME
        self.writer = DeepSeekCodeWriter(self.model_name)
