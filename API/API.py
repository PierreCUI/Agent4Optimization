import base64
import json
import os
import warnings
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from configs import IMAGE_SAMPLE_HEAD_TAIL, IMAGE_SAMPLE_MIDDLE, MAX_IMAGES_PER_REQUEST
from history import init_log, record as _record_log

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


def _cap_images(
    images: list[str],
    max_images: int = MAX_IMAGES_PER_REQUEST,
    head_tail: int = IMAGE_SAMPLE_HEAD_TAIL,
    middle: int = IMAGE_SAMPLE_MIDDLE,
) -> list[str]:
    """
    Cap *images* at *max_images* images for one multimodal request.

    When a run produced more snapshots than fit in a single request
    (more than ``max_images``), keep the first ``head_tail``, the last
    ``head_tail``, and ``middle`` evenly spaced picks from between
    them — the earliest and latest snapshots are always present, and
    the middle history stays representative.
    """
    n = len(images)
    if n <= max_images:
        return list(images)
    if head_tail > 0 and 2 * head_tail >= max_images:
        # The cap cannot hold the full head+tail — keep BOTH ends anyway
        # (earliest AND latest snapshots are the documented guarantee),
        # shrinking them to fit.
        half = max_images // 2
        return list(images[:half]) + list(images[-(max_images - half):])
    head = list(images[:head_tail]) if head_tail > 0 else []
    tail = list(images[-head_tail:]) if head_tail > 0 else []
    mid = list(images[head_tail:-head_tail]) if head_tail > 0 else list(images)
    m = len(mid)
    if m == 0 or middle <= 0:
        picked = []
    elif m == 1 or middle == 1:
        picked = mid[:1]
    else:
        cnt = min(middle, max_images - 2 * head_tail, m)
        picked = [mid[(k * (m - 1)) // (cnt - 1)] for k in range(cnt)]
    # cnt is bounded by max_images - 2*head_tail, so the cap holds for
    # ANY configuration without ever slicing off the tail.
    return head + picked + tail


class CodeWriter:
    def __init__(self):
        self.model_name = None
        self._model = None
        self._prompt = None
        self._parser = StrOutputParser()
        self._chain = None

    def initialization(self, prompt_list=None):
        self._prompt = ChatPromptTemplate.from_messages(prompt_list)
        self._chain = self._prompt | self._model

    def invoke(self, function_dict):
        response = self._chain.invoke(function_dict)
        self._report_usage(response)
        return self._parser.invoke(response)

    def invoke_multimodal(self, function_dict, images=None):
        """Like invoke(), but appends *images* (PNG file paths, rendered by
        the reflection loop) as image_url blocks to the last message.

        Uses the OpenAI-compatible multimodal format (data-URL base64),
        supported by the deepseek-v4-*-vision models via
        ``ChatOpenAI``-compatible endpoints.  The last entry of
        ``prompt_list`` must be the user message; its text is rendered
        through the same ChatPromptTemplate (``function_dict`` gets
        ``image_section`` defaulted to ``''`` so non-vision callers keep
        working).
        """
        function_dict = dict(function_dict)
        # The template may carry an {image_section} placeholder even when
        # no image is attached — default it on EVERY branch so the
        # no-image fallback cannot crash on a missing variable.
        function_dict.setdefault("image_section", "")

        if not images:
            return self.invoke(function_dict)

        images = _cap_images(images)
        messages = self._prompt.format_messages(**function_dict)

        text = messages[-1].content
        if not isinstance(text, str):
            text = str(text)
        blocks = [{"type": "text", "text": text}]
        for img_path in images:
            try:
                b64 = base64.b64encode(Path(img_path).read_bytes()).decode()
            except OSError as e:
                warnings.warn(f"invoke_multimodal: cannot read image {img_path}: {e}")
                continue
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        messages[-1] = HumanMessage(content=blocks)
        # bypass the string-formatting chain: multimodality needs the
        # raw message list (a ChatPromptTemplate renders content=str()
        # which would mangle the image blocks).
        response = self._model.invoke(messages)
        self._report_usage(response)
        return self._parser.invoke(response)

    def _report_usage(self, response):
        """Print token usage of *response* (shared by invoke paths)."""
        try:
            # langchain-core ≥1.5 exposes AIMessage.usage_metadata as a
            # property; OpenAI-compatible providers never put the counts
            # under response_metadata['input_tokens'] — only this branch
            # is correct.
            print(
                f"Usage: "
                f"Input {response.usage_metadata['input_tokens']} "
                f"Output {response.usage_metadata['output_tokens']}."
            )
        except (KeyError, TypeError, AttributeError) as e:
            warnings.warn(f"Failed to parse token usage from response metadata: {e}")


class CodeAgent:
    def __init__(self):
        self.model_name = None
        self.writer = None
        self.prompt_list = []
        self.history_log = None  # Path to results/<problem>/history.txt
        self.history_label = ""  # stage/round context for every record

    # ── operation log (results/<problem>/history.txt) ──────────────

    def attach_history(self, history_path, write_header=True):
        """Redirect the operation log to *history_path*.

        Called once per problem from main.py; every subsequent
        invoke/invoke_multimodal appends one record (time, kind, label,
        FULL prompt + input values + output) to the file.

        ``write_header=False`` only points the log (used by the router's
        vision agent and the reflection probe agents — the header is
        created once, and a second write would truncate the records).
        """
        self.history_log = Path(history_path)
        if not write_header:
            return
        init_log(self.history_log, self.history_log.parent.name)

    def _record_history(self, kind: str, full_text: str, function_dict=None, images=None) -> None:
        """Append one record to history.txt (best effort, full content).

        Records: the operation kind, the stage/round label, the COMPLETE
        prompt templates, the values actually passed (function_dict),
        attached image names, and the model output — the log is a
        complete record of the pipeline's work and may grow large by
        design (no truncation).
        """
        if self.history_log is None:
            return
        prompt_text = ""
        for m in self.prompt_list or []:
            if isinstance(m, dict) and m.get("content"):
                prompt_text += (
                    f"\n--- {m.get('role', '?')} prompt ---\n"
                    f"{m.get('content', '')}\n"
                )
        # default=str: some callers pass small non-serializable values
        # (Path, sets) — the log must never break an invoke.
        values = json.dumps(function_dict or {}, ensure_ascii=False, indent=2, default=str)
        input_text = prompt_text + "\n--- input values ---\n" + values
        _record_log(
            self.history_log,
            kind,
            label=self.history_label,
            input_text=input_text,
            output_text=str(full_text),
            images=images,
        )

    def invoke(self, function_dict):
        self.writer.initialization(self.prompt_list)
        full_text = self.writer.invoke(function_dict)
        self._record_history("LLM text", full_text, function_dict)
        return full_text

    def invoke_multimodal(self, function_dict, images=None):
        """invoke() with vision support: *images* are PNG paths appended
        as image_url blocks to the last (user) message; the model sees
        them and answers with text (used by the reflection loop)."""
        self.writer.initialization(self.prompt_list)
        full_text = self.writer.invoke_multimodal(function_dict, images=images)
        # No images is just a text call (the writer falls back) — record
        # it as "LLM text", not "LLM vision".
        kind = "LLM vision" if images else "LLM text"
        self._record_history(kind, full_text, function_dict, images=images)
        return full_text

    def set_prompt_list(self, prompt_list):
        self.prompt_list = prompt_list


def test_connection(agent, attempts: int = 2) -> bool:
    """Minimal LLM round-trip connectivity test.

    Used at the START of the pipeline and of the reflection loop: an
    unreachable model ends the run immediately instead of burning
    repair rounds on connection errors.
    """
    agent.set_prompt_list([
        {"role": "user", "content": "Reply with exactly one word: OK"},
    ])
    for _ in range(max(1, attempts)):
        try:
            raw = str(agent.invoke({})).strip()
            if raw:
                return True
        except Exception:
            continue
    return False


class AgentRouter:
    """
    Routes every LLM call to one of two underlying agents:

    - text-only calls (``invoke``) → *text_agent*
    - multimodal calls (``invoke_multimodal`` with images) → *vision_agent*

    Both must expose the CodeAgent interface.  ``set_prompt_list`` and
    ``attach_history`` are forwarded to each so both stay in sync; the
    shared log records text and vision calls in one chronological file.
    """

    def __init__(self, text_agent, vision_agent):
        self.text_agent = text_agent
        self.vision_agent = vision_agent

    def set_prompt_list(self, prompt_list):
        self.text_agent.set_prompt_list(prompt_list)
        self.vision_agent.set_prompt_list(prompt_list)

    def attach_history(self, history_path):
        # Text agent creates/truncates the log + writes the header; the
        # vision agent only points at it (no duplicate header, no
        # truncation of already-appended records).
        self.text_agent.attach_history(history_path, write_header=True)
        self.vision_agent.attach_history(history_path, write_header=False)

    @property
    def history_label(self) -> str:
        """The stage/round label both underlying agents record."""
        return self.vision_agent.history_label

    @history_label.setter
    def history_label(self, value: str) -> None:
        # set on BOTH — either agent may handle the next call
        self.text_agent.history_label = value
        self.vision_agent.history_label = value

    def invoke(self, function_dict):
        return self.text_agent.invoke(function_dict)

    def invoke_multimodal(self, function_dict, images=None):
        return self.vision_agent.invoke_multimodal(
            function_dict, images=images
        )
