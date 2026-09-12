import base64
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from configs import IMAGE_SAMPLE_HEAD_TAIL, IMAGE_SAMPLE_MIDDLE, MAX_IMAGES_PER_REQUEST

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
    _HISTORY_TEXT_LIMIT = 20_000  # chars per input/output record

    def __init__(self):
        self.model_name = None
        self.writer = None
        self.prompt_list = []
        self.history_log = None  # Path to results/<problem>/history.txt

    # ── operation log (results/<problem>/history.txt) ──────────────

    def attach_history(self, history_path, write_header=True):
        """Redirect the operation log to *history_path*.

        Called once per problem from main.py; every subsequent
        invoke/invoke_multimodal appends one record (time, operation,
        input, output) to the file.

        ``write_header=False`` only points the log (used by the router's
        vision agent — the text agent already created the file and wrote
        the header, so a second write would both duplicate the header
        and truncate the records).
        """
        self.history_log = Path(history_path)
        if not write_header:
            return
        try:
            self.history_log.parent.mkdir(parents=True, exist_ok=True)
            # "w": one pipeline run = one log.  Re-running the same
            # problem starts fresh (like reflection.txt) instead of
            # accumulating stale records from the previous run.
            with open(self.history_log, "w", encoding="utf-8") as f:
                f.write(f"{'=' * 70}\n")
                f.write(f"OPERATION LOG — {self.history_log.parent.name}\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'=' * 70}\n")
        except OSError as e:
            warnings.warn(f"history log init failed: {e}")
            self.history_log = None

    def _system_hint(self) -> str:
        """First line of the system prompt — names the operation."""
        for m in self.prompt_list:
            if isinstance(m, dict) and m.get("role") == "system":
                return " ".join(str(m.get("content", "")).split())[:160]
        return ""

    def _record_history(self, kind: str, full_text: str, function_dict=None, images=None) -> None:
        """Append one human-readable record to history.txt (best effort).

        Records: the operation kind, the prompt template, the values
        actually passed (function_dict), attached image names, and the
        model output.
        """
        if self.history_log is None:
            return
        try:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # default=str: some callers pass small non-serializable values
            # (Path, sets) — the log must never break an invoke.
            values = json.dumps(function_dict or {}, ensure_ascii=False, indent=2, default=str)
            if len(values) > self._HISTORY_TEXT_LIMIT:
                values = values[: self._HISTORY_TEXT_LIMIT] + "\n... [truncated]"
            out = str(full_text)
            if len(out) > self._HISTORY_TEXT_LIMIT:
                out = out[: self._HISTORY_TEXT_LIMIT] + "\n... [truncated]"
            with open(self.history_log, "a", encoding="utf-8") as f:
                f.write(f"\n=== {stamp} | {kind} | {self._system_hint()} ===\n")
                f.write("--- input values ---\n")
                f.write(values + "\n")
                if images:
                    names = [Path(p).name for p in images[:5]]
                    f.write(
                        f"--- images ---\n{len(images)} image(s): {names}"
                        f"{' ...' if len(images) > 5 else ''}\n"
                    )
                f.write("--- output ---\n")
                f.write(out + "\n")
        except (OSError, TypeError) as e:
            warnings.warn(f"history log write failed: {e}")

    def invoke(self, function_dict):
        self.writer.initialization(self.prompt_list)
        full_text = self.writer.invoke(function_dict)
        self._record_history("text", full_text, function_dict)
        return full_text

    def invoke_multimodal(self, function_dict, images=None):
        """invoke() with vision support: *images* are PNG paths appended
        as image_url blocks to the last (user) message; the model sees
        them and answers with text (used by the reflection loop)."""
        self.writer.initialization(self.prompt_list)
        full_text = self.writer.invoke_multimodal(function_dict, images=images)
        # No images is just a text call (the writer falls back) — record
        # it as "text", not "vision".
        kind = "vision" if images else "text"
        self._record_history(kind, full_text, function_dict, images=images)
        return full_text

    def set_prompt_list(self, prompt_list):
        self.prompt_list = prompt_list


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

    def invoke(self, function_dict):
        return self.text_agent.invoke(function_dict)

    def invoke_multimodal(self, function_dict, images=None):
        return self.vision_agent.invoke_multimodal(
            function_dict, images=images
        )
