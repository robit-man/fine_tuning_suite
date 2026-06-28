# Fine-Tuning Suite

End-to-end pipeline for LLM fine-tuning, tool splicing, GGUF export, and Ollama registry publishing.  
Supports Qwen-based models from 9B to 397B with **vision, tools, thinking, instruction following**.

Includes a **dark-theme Flask dashboard** with **RESTful API** for agent/MCP toolkit integration,  
and a `tool_splice.py` pipeline for importing HuggingFace models -> Ollama with tool-calling config.

## Quick Start — Training Pipeline

```bash
python app.py bootstrap
python curate_r7.py
DISTILL_TRAIN_FILE=r7_additive_train DISTILL_VAL_FILE=r7_additive_val \
  DISTILL_OUTPUT_SUFFIX=.r7-additive DISTILL_LR=1e-4 \
  DISTILL_LORA_R=32 DISTILL_LORA_ALPHA=64 \
  DISTILL_EPOCHS=1 DISTILL_PATIENCE=3 \
  python app.py train
python splice_and_export.sh
python eval_diverse.py <model_name> --base qwen3.5:9b
```

## Dashboard — Dark Theme + REST API

```bash
pip install flask jinja2 werkzeug httpx
python -m training_suite db-init
python -m training_suite web --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860` — dark GitHub-themed UI with card-based layout.

### RESTful API (Agent/MCP Toolkit)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/models` | List all models |
| `POST` | `/api/models` | Intake a model (`{source, target_capabilities}`) |
| `GET` | `/api/models/<id>` | Model detail |
| `GET` | `/api/jobs` | List jobs (`?limit=N`) |
| `POST` | `/api/jobs` | Start job (`{action, model_id, dataset_id}`) |
| `GET` | `/api/jobs/<id>` | Job detail + live log |
| `POST` | `/api/jobs/<id>/cancel` | Cancel job |
| `GET` | `/api/datasets` | List datasets |
| `POST` | `/api/datasets` | Register dataset |
| `GET` | `/api/evals` | List eval runs |
| `POST` | `/api/evals` | Run evaluation |
| `GET` | `/api/actions` | Available actions/commands |
| `GET` | `/api/ollama/show?model=<tag>` | Ollama model details |

## Quick Start — Tool Splice & Ollama Upload

```bash
python tool_splice.py 9b     # Ornith-1.0-9B into Ollama
python tool_splice.py 35b    # Ornith-1.0-35B into Ollama
python tool_splice.py both   # Both sizes
```

## File Index

| File | Purpose |
|------|---------|
| `web.py` | Flask app: dark theme UI + RESTful API |
| `static/styles.css` | GitHub-dark theme CSS |
| `templates/*.html` | Card-based Jinja2 templates |
| `tool_splice.py` | HuggingFace -> Ollama import pipeline |
| `core/state.py` | SQLite state store |
| `core/jobs.py` | Background subprocess job runner |
| `core/config.py` | Paths, constants, utilities |
| `models/ollama.py` | Ollama show, Modelfile, create/push |
| `models/intake.py` | HuggingFace model inspection |
| `models/gguf.py` | GGUF file inspection |
| `training/adapters.py` | Action specifications |
| `evals/runner.py` | Capability gate, tool smoke, eval specs |
| `cli.py` | CLI entry point |
| `app.py` | Training harness |
| `splice_vision_v2.py` | Vision-splice merged model builder |

## Models on Ollama Registry

| Tag | Size | Capabilities |
|-----|------|-------------|
| `robit/ornith:9b` | 5.6 GB | tools, thinking, completion |
| `robit/ornith:35b` | 21 GB | tools, thinking, completion |

See [AGENTS.md](AGENTS.md) for full agent instructions.
