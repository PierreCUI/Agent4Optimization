# Agent4FreeFem

An LLM-agent pipeline that turns natural-language problem descriptions into working
topology-optimization code for [AutoFreeFEM](https://gitlab.tugraz.at/autofreefem/autofreefem)
and FreeFem++, then executes, diagnoses, and repairs it automatically.

## Pipeline

Each demo problem in `demos/` is processed end-to-end:

1. **Safety check** — the LLM screens the problem description.
2. **Example selection** — an LLM picks relevant reference examples from `database/`.
3. **Initialization** — spec generation: a `ProblemSpec` and a mesh spec are produced
   (with a generate → review → repair loop) and saved as JSON.
4. **Generation** — the LLM writes `main.py` (AutoFreeFEM code) from the spec, the
   mesh, and the selected examples.
5. **Static check & repair** — a check → review → repair loop fixes static issues,
   followed by normalization of designed tunables.
6. **Execution** — the code runs under AutoFreeFEM with a FreeFem++ parameter sweep
   (up to 64 cells, 16 concurrent workers).
7. **Reflection** — failures are diagnosed from FreeFEM console output and design
   snapshots (a vision model); fixes are applied and the code re-executed, for up to
   10 repair rounds.

All LLM traffic, repair histories, and per-case artifacts are logged under
`results/<problem>/`.

## Repository layout

| Path              | Purpose                                              |
|-------------------|------------------------------------------------------|
| `API/`            | LLM wrappers (DeepSeek, ChatGPT, Gemini, Qwen) and a text/vision model router |
| `Initialization/` | Safety check, example selection, spec + mesh generation |
| `Generation/`     | Code generation, static checks, design-parameter normalization |
| `Execution/`      | Runner: AutoFreeFEM + FreeFem++ sweep (Windows/Linux) |
| `Reflection/`     | Error classification, repair, rendering, serial reflection loop |
| `database/`       | Reference FreeFEM examples and functions             |
| `demos/`          | Benchmark problems (beams, brackets, heat sinks, cooling plates, …) |
| `configs.py`      | Global pipeline constants                            |
| `main.py`         | Pipeline orchestration                               |

## Requirements

- Python 3.11 (the pipeline manages a conda environment named `autofreefem`)
- [AutoFreeFEM](https://gitlab.tugraz.at/autofreefem/autofreefem) (installed from
  `requirements.txt`)
- FreeFem++ (freefem / ff-mpirun) for the solver sweep
- LLM API keys configured in local `.env` files (not tracked by git)

## Usage

```bash
python main.py
```

This initializes the example database once and runs the full pipeline on every
demo problem in `demos/`. For a single problem:

```python
from main import main
from API import AgentRouter, DeepSeekCodeAgentV4Pro, DeepSeekCodeAgentV4FlashVision

agent = AgentRouter(DeepSeekCodeAgentV4Pro(), DeepSeekCodeAgentV4FlashVision())
main(agent, "run_mbbBeam")
```

## License

[Apache License 2.0](LICENSE)
