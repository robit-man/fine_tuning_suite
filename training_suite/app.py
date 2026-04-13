#!/usr/bin/env python3
"""
Qwen3.5-9B Reasoning Distillation Harness
==========================================

End-to-end pipeline that distills reasoning traces from
  khazarai/qwen3.6-plus-high-reasoning-500x  (teacher: Qwen3.6-plus)
into
  Qwen/Qwen3.5-9B  (student)
via LoRA SFT, then exports to GGUF and registers with Ollama.

Single-file, self-bootstrapping. On first run it creates a .venv with uv,
installs pinned deps, and re-execs itself inside the venv.

Usage
-----
  python app.py bootstrap         # Just create the venv
  python app.py prepare           # Download dataset, build train/val/test splits
  python app.py baseline          # Evaluate BASE model on held-out test + GSM8K
  python app.py train             # Run LoRA SFT across available GPUs (DDP)
  python app.py eval              # Evaluate TUNED model, produce delta report
  python app.py export            # Merge LoRA, convert to GGUF, create Ollama model
  python app.py all               # Run everything in order

Anti-reward-hack rules enforced in code
---------------------------------------
  * Test split is written to disk with a fixed seed and NEVER seen during train
  * Val loss is computed on a separate held-out split
  * Baseline eval runs on the UNMODIFIED base model first
  * Post-train eval uses the SAME test samples + SAME GSM8K sample IDs
  * Report prints delta (baseline vs tuned) and flags suspicious overfitting
  * Training loss is masked so it's computed only on assistant tokens
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Section 0: Project constants (pure stdlib — safe to import before venv)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQS = ROOT / "requirements.txt"
MARKER = VENV / ".installed.sha256"

# GPU selection. User first said use 2 (0,2) while ollama held GPU 1, then later
# freed GPU 1 — so we use all 3 A100s from training onward. Baseline already ran
# on 0,2 and that's fine: no state leakage since we only read, not write.
# Override via env DISTILL_GPUS="0,1,2" etc.
CUDA_VISIBLE = os.environ.get("DISTILL_GPUS", "0,1,2")
NUM_GPUS = len(CUDA_VISIBLE.split(","))

CONFIG = {
    "base_model": "Qwen/Qwen3.5-9B",
    "dataset_id": "khazarai/qwen3.6-plus-high-reasoning-500x",
    "dataset_file": "Qwen3.6-plus.jsonl",
    "output_name": "qwen3.5-9b-qwen3.6-reasoning-distilled",
    "seed": 42,
    "splits": {"train": 0.85, "val": 0.075, "test": 0.075},
    "train": {
        "per_device_batch_size": 1,
        "grad_accum_steps": 8,
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        # Lowered from 8192 to cap activation memory on pathologically long
        # samples in the Crownelius dataset (max total chars 28K ≈ 8K tokens);
        # Crownelius median is 2588 chars so truncation affects only the tail
        "max_seq_length": 5120,
        "weight_decay": 0.01,
        "logging_steps": 2,
        "eval_strategy": "steps",
        "eval_steps": 5,
        "save_strategy": "steps",
        "save_steps": 5,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "early_stopping_patience": 3,
    },
    "lora": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.05,
        # "all-linear" lets PEFT auto-detect; excludes embeddings & lm_head
        "target_modules": "all-linear",
    },
    "eval": {
        "gsm8k_n": 100,
        "max_new_tokens": 1024,
        "temperature": 0.0,  # greedy for reproducibility
        "gsm8k_seed": 1337,
    },
}

PATHS = {
    "data": ROOT / "data",
    "splits": ROOT / "data" / "splits",
    "gsm8k": ROOT / "data" / "gsm8k_sample.jsonl",
    "tool_bench_dir": ROOT / "data" / "tool_bench",
    "tool_bench": ROOT / "data" / "tool_bench" / "bench.jsonl",
    "tool_defs": ROOT / "data" / "tool_bench" / "tools.json",
    "tool_manifest": ROOT / "data" / "tool_bench" / "manifest.json",
    "checkpoints": ROOT / "outputs" / "checkpoints",
    "merged": ROOT / "outputs" / "merged",
    "gguf": ROOT / "outputs" / "gguf",
    "reports": ROOT / "outputs" / "reports",
    "logs": ROOT / "logs",
    "llama_cpp": ROOT / "vendor" / "llama.cpp",
}

# Ollama endpoint for tool-call benchmark (verify-ollama + baseline_tools + eval_tools)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_BASE_MODEL = "qwen3.5:9b"
OLLAMA_TUNED_MODEL = "qwen3.5-9b-qwen3.6-distilled:q4km"

# ---------------------------------------------------------------------------
# Section 1: Bootstrap (venv + deps)
# ---------------------------------------------------------------------------


def _reqs_hash() -> str:
    return hashlib.sha256(REQS.read_bytes()).hexdigest()


def _in_project_venv() -> bool:
    try:
        return Path(sys.prefix).resolve() == VENV.resolve()
    except Exception:
        return False


def bootstrap(force: bool = False) -> None:
    """Create .venv via uv (fallback stdlib), install requirements, re-exec self."""
    if _in_project_venv() and MARKER.exists() and MARKER.read_text().strip() == _reqs_hash():
        return  # Already set up and matches requirements hash

    print(f"[bootstrap] Project root: {ROOT}")
    print(f"[bootstrap] Target venv:  {VENV}")

    have_uv = shutil.which("uv") is not None
    if not VENV.exists() or force:
        if VENV.exists() and force:
            shutil.rmtree(VENV)
        if have_uv:
            print("[bootstrap] Creating venv with uv...")
            subprocess.check_call(["uv", "venv", str(VENV), "--python", "3.12"])
        else:
            print("[bootstrap] Creating venv with stdlib venv (uv not found)...")
            subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])

    vpy = VENV / "bin" / "python"
    if not vpy.exists():
        raise RuntimeError(f"venv python missing at {vpy}")

    # Install deps if hash mismatch or marker absent
    current = _reqs_hash()
    needs_install = not MARKER.exists() or MARKER.read_text().strip() != current
    if needs_install:
        print("[bootstrap] Installing requirements...")
        if have_uv:
            subprocess.check_call(
                ["uv", "pip", "install", "--python", str(vpy), "-r", str(REQS)]
            )
        else:
            subprocess.check_call([str(vpy), "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.check_call([str(vpy), "-m", "pip", "install", "-r", str(REQS)])
        MARKER.write_text(current)
        print("[bootstrap] Requirements installed; marker written.")
    else:
        print("[bootstrap] Requirements already installed (hash match).")

    # Re-exec self inside venv preserving args
    if not _in_project_venv():
        print(f"[bootstrap] Re-exec into {vpy}")
        os.execv(str(vpy), [str(vpy), str(Path(__file__).resolve()), *sys.argv[1:]])


# ---------------------------------------------------------------------------
# Section 2: Environment + logging helpers (imported after bootstrap)
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    for p in PATHS.values():
        if p.suffix:  # file path — ensure parent
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)


def _log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def _set_gpus() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Section 3: Data preparation
# ---------------------------------------------------------------------------


def cmd_prepare() -> None:
    """Download dataset, build deterministic train/val/test splits, sample GSM8K."""
    _ensure_dirs()
    _set_gpus()
    import random

    from datasets import load_dataset

    _log("prepare", f"Loading {CONFIG['dataset_id']}...")
    ds = load_dataset(CONFIG["dataset_id"], split="train")
    _log("prepare", f"Loaded {len(ds)} rows. Columns: {ds.column_names}")

    # Deterministic shuffle + split. We write splits to disk so train NEVER
    # re-shuffles. This is an anti-reward-hack guarantee: test is frozen.
    indices = list(range(len(ds)))
    rng = random.Random(CONFIG["seed"])
    rng.shuffle(indices)

    n_total = len(ds)
    n_train = int(n_total * CONFIG["splits"]["train"])
    n_val = int(n_total * CONFIG["splits"]["val"])
    n_test = n_total - n_train - n_val  # absorb rounding into test
    _log("prepare", f"Split sizes: train={n_train} val={n_val} test={n_test}")

    train_idx = sorted(indices[:n_train])
    val_idx = sorted(indices[n_train : n_train + n_val])
    test_idx = sorted(indices[n_train + n_val :])
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    splits_dir = PATHS["splits"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    def _dump(name: str, idxs: list[int]) -> None:
        out = splits_dir / f"{name}.jsonl"
        with out.open("w") as f:
            for i in idxs:
                row = ds[int(i)]
                f.write(json.dumps({"messages": row["messages"], "orig_index": int(i)}) + "\n")
        _log("prepare", f"  wrote {out.name} ({len(idxs)} rows)")

    _dump("train", train_idx)
    _dump("val", val_idx)
    _dump("test", test_idx)

    # Manifest with index lists for auditability
    manifest = {
        "dataset_id": CONFIG["dataset_id"],
        "seed": CONFIG["seed"],
        "n_total": n_total,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "sha256_train": _file_sha(splits_dir / "train.jsonl"),
        "sha256_val": _file_sha(splits_dir / "val.jsonl"),
        "sha256_test": _file_sha(splits_dir / "test.jsonl"),
    }
    (splits_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _log("prepare", f"Manifest written to {splits_dir / 'manifest.json'}")

    # GSM8K sample for external honesty probe
    _prepare_gsm8k()


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_gsm8k() -> None:
    from datasets import load_dataset

    _log("prepare", "Sampling GSM8K (external honesty probe)...")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    # Fixed-seed sample of N questions
    import random

    rng = random.Random(CONFIG["eval"]["gsm8k_seed"])
    idx = rng.sample(range(len(gsm)), CONFIG["eval"]["gsm8k_n"])
    with PATHS["gsm8k"].open("w") as f:
        for i in sorted(idx):
            row = gsm[int(i)]
            f.write(json.dumps({"question": row["question"], "answer": row["answer"], "idx": int(i)}) + "\n")
    _log("prepare", f"  wrote {PATHS['gsm8k']} ({CONFIG['eval']['gsm8k_n']} samples)")


# ---------------------------------------------------------------------------
# Section 4: Model loading helpers
# ---------------------------------------------------------------------------


def _load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _load_base_model(dtype: str = "bf16"):
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    _log("model", f"Loading base model {CONFIG['base_model']} (dtype={dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    return model


# ---------------------------------------------------------------------------
# Section 5: Baseline evaluation (run BEFORE any training)
# ---------------------------------------------------------------------------


def cmd_baseline() -> None:
    """Evaluate untouched base model on held-out test + GSM8K sample."""
    _ensure_dirs()
    _set_gpus()
    _log("baseline", "Evaluating BASE model (pre-training floor)...")
    results = _evaluate_model(model_kind="base", adapter_path=None)
    out = PATHS["reports"] / "baseline.json"
    out.write_text(json.dumps(results, indent=2))
    _log("baseline", f"Saved {out}")
    _print_eval_summary("BASELINE", results)


def _evaluate_model(model_kind: str, adapter_path: Path | None):
    """Teacher-forced loss on held-out test + GSM8K greedy accuracy."""
    import torch
    from transformers import AutoModelForCausalLM

    tok = _load_tokenizer()
    _log("eval", f"Loading {model_kind} model...")
    model = _load_base_model()
    if adapter_path is not None:
        from peft import PeftModel

        _log("eval", f"Attaching LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    # --- Part A: teacher-forced loss on test split (proxy for distillation fit)
    test_loss = _teacher_forced_loss(model, tok, PATHS["splits"] / "test.jsonl")
    _log("eval", f"  test_split_loss = {test_loss:.4f}")

    # --- Part B: GSM8K greedy accuracy (external reasoning probe)
    gsm_results = _gsm8k_eval(model, tok)
    _log("eval", f"  gsm8k_accuracy  = {gsm_results['accuracy']:.3f} ({gsm_results['correct']}/{gsm_results['total']})")

    del model
    torch.cuda.empty_cache()

    return {
        "model_kind": model_kind,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "test_split_loss": test_loss,
        "test_split_perplexity": float(torch.tensor(test_loss).exp().item()),
        "gsm8k": gsm_results,
    }


def _apply_chat_template_sample(tok, messages: list[dict]) -> str:
    """Apply Qwen chat template WITHOUT generation prompt, preserving <think> blocks."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _teacher_forced_loss(model, tok, path: Path) -> float:
    """Compute average per-token NLL on assistant tokens of held-out samples."""
    import torch

    total_loss = 0.0
    total_tokens = 0
    max_len = CONFIG["train"]["max_seq_length"]

    with path.open() as f:
        for line in f:
            row = json.loads(line)
            messages = row["messages"]
            # Build full text + prefix-only text to locate assistant span
            full_text = _apply_chat_template_sample(tok, messages)
            user_only = tok.apply_chat_template(
                [m for m in messages if m["role"] == "user"],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_ids = tok(full_text, return_tensors="pt", truncation=True, max_length=max_len).input_ids
            prefix_ids = tok(user_only, return_tensors="pt", truncation=True, max_length=max_len).input_ids
            if full_ids.shape[1] <= prefix_ids.shape[1]:
                continue
            full_ids = full_ids.to(model.device)
            with torch.no_grad():
                out = model(full_ids)
                logits = out.logits[:, :-1, :]
                targets = full_ids[:, 1:]
                # Mask everything before assistant region
                asst_start = prefix_ids.shape[1] - 1
                mask = torch.zeros_like(targets, dtype=torch.bool)
                mask[:, asst_start:] = True
                # CE per token on masked region
                loss_per_tok = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape_as(targets)
                loss_sum = (loss_per_tok * mask.float()).sum().item()
                n_tok = int(mask.sum().item())
                if n_tok > 0:
                    total_loss += loss_sum
                    total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


def _gsm8k_eval(model, tok) -> dict:
    """Greedy decode + numeric answer extraction on locked GSM8K sample."""
    import re
    import torch

    results = {"correct": 0, "total": 0, "items": []}
    max_new = CONFIG["eval"]["max_new_tokens"]
    ans_re = re.compile(r"-?\$?\d+(?:,\d{3})*(?:\.\d+)?")

    with PATHS["gsm8k"].open() as f:
        for line in f:
            row = json.loads(line)
            q = row["question"]
            gold_raw = row["answer"].split("####")[-1].strip().replace(",", "")
            try:
                gold = float(gold_raw)
            except ValueError:
                continue
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Solve this math problem. Show your reasoning, then give the "
                        f"final answer on its own line as: Answer: <number>\n\n{q}"
                    ),
                }
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            # Extract answer: prefer last "Answer: X" match, else last number
            m = re.findall(r"[Aa]nswer\s*:?\s*(-?\$?\d[\d,]*\.?\d*)", gen)
            if m:
                raw = m[-1]
            else:
                nums = ans_re.findall(gen)
                raw = nums[-1] if nums else ""
            raw = raw.replace("$", "").replace(",", "")
            try:
                pred = float(raw)
                correct = abs(pred - gold) < 1e-4
            except ValueError:
                pred = None
                correct = False
            results["total"] += 1
            results["correct"] += int(correct)
            results["items"].append(
                {"idx": row["idx"], "gold": gold, "pred": pred, "correct": correct}
            )
    results["accuracy"] = results["correct"] / max(results["total"], 1)
    return results


def _print_eval_summary(tag: str, r: dict) -> None:
    print(f"\n===== {tag} =====")
    print(f"  test_split_loss       : {r['test_split_loss']:.4f}")
    print(f"  test_split_perplexity : {r['test_split_perplexity']:.2f}")
    print(f"  gsm8k_accuracy        : {r['gsm8k']['accuracy']:.3f}  ({r['gsm8k']['correct']}/{r['gsm8k']['total']})")
    print("===============\n")


# ---------------------------------------------------------------------------
# Section 6: Training (LoRA SFT via TRL, launched with torchrun for DDP)
# ---------------------------------------------------------------------------


def cmd_train() -> None:
    """Launch DDP training via torchrun on GPUs 0,2 (NUM_GPUS processes)."""
    _ensure_dirs()
    _set_gpus()
    # Relaunch under torchrun so we get proper DDP with 2 processes
    if os.environ.get("_TORCHRUN_STARTED") != "1":
        vpy = VENV / "bin" / "python"
        torchrun = VENV / "bin" / "torchrun"
        cmd = [
            str(torchrun),
            f"--nproc_per_node={NUM_GPUS}",
            "--master_port=29501",
            str(Path(__file__).resolve()),
            "_train_worker",
        ]
        env = os.environ.copy()
        env["_TORCHRUN_STARTED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE
        _log("train", f"Launching: {' '.join(shlex.quote(x) for x in cmd)}")
        proc = subprocess.run(cmd, env=env)
        sys.exit(proc.returncode)
    else:
        _train_worker()


def _train_worker() -> None:
    """Actual training logic — invoked per-rank via torchrun."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, EarlyStoppingCallback
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    tok = _load_tokenizer()

    def _load_split(name: str):
        rows = []
        with (PATHS["splits"] / f"{name}.jsonl").open() as f:
            for line in f:
                rows.append(json.loads(line))
        return Dataset.from_list(rows)

    # DISTILL_TRAIN_FILE / DISTILL_VAL_FILE override the split filenames
    # (without .jsonl) so we can run additional training rounds on different
    # datasets without touching the original frozen splits.
    train_name = os.environ.get("DISTILL_TRAIN_FILE", "train")
    val_name = os.environ.get("DISTILL_VAL_FILE", "val")
    train_ds = _load_split(train_name)
    val_ds = _load_split(val_name)
    _log("train", f"train=[{train_name}] {len(train_ds)}  val=[{val_name}] {len(val_ds)}")

    def _format(example):
        # messages → chat-template string; TRL will tokenize+mask
        return {"text": _apply_chat_template_sample(tok, example["messages"])}

    # Remove all columns except "text" (which _format adds). Different rounds
    # may have different columns (orig_index, category, etc.) so we just drop
    # everything the dataset came with.
    train_cols = [c for c in train_ds.column_names if c != "text"]
    val_cols = [c for c in val_ds.column_names if c != "text"]
    train_ds = train_ds.map(_format, remove_columns=train_cols)
    val_ds = val_ds.map(_format, remove_columns=val_cols)

    _log("train", f"Loading base model {CONFIG['base_model']} ...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Allow env-var overrides for hyperparameter tuning across rounds.
    # R4 uses lower LR + lower rank to reduce catastrophic forgetting.
    lora_r = int(os.environ.get("DISTILL_LORA_R", CONFIG["lora"]["r"]))
    lora_alpha = int(os.environ.get("DISTILL_LORA_ALPHA", CONFIG["lora"]["alpha"]))
    _log("train", f"LoRA r={lora_r} alpha={lora_alpha}")
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=CONFIG["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=CONFIG["lora"]["target_modules"],
    )

    tcfg = CONFIG["train"]
    # DISTILL_OUTPUT_SUFFIX appends to the checkpoint dir name so we can run
    # additional training rounds without clobbering the first-round adapter.
    suffix = os.environ.get("DISTILL_OUTPUT_SUFFIX", "")
    out_dir = PATHS["checkpoints"] / (CONFIG["output_name"] + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allow env-var override for learning rate (R4 uses 5e-5 instead of 2e-4)
    lr = float(os.environ.get("DISTILL_LR", tcfg["learning_rate"]))
    num_epochs = int(os.environ.get("DISTILL_EPOCHS", tcfg["num_epochs"]))
    patience = int(os.environ.get("DISTILL_PATIENCE", tcfg["early_stopping_patience"]))
    _log("train", f"lr={lr} epochs={num_epochs} patience={patience}")

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum_steps"],
        learning_rate=lr,
        warmup_ratio=tcfg["warmup_ratio"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        weight_decay=tcfg["weight_decay"],
        max_seq_length=tcfg["max_seq_length"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=tcfg["logging_steps"],
        eval_strategy=tcfg["eval_strategy"],
        eval_steps=tcfg["eval_steps"],
        save_strategy=tcfg["save_strategy"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        greater_is_better=tcfg["greater_is_better"],
        report_to="none",
        seed=CONFIG["seed"],
        data_seed=CONFIG["seed"],
        dataset_text_field="text",
        packing=False,
        ddp_find_unused_parameters=False,
    )

    # Completion-only loss: mask everything before the assistant turn marker.
    # For Qwen3.5 chat template, assistant turns begin with "<|im_start|>assistant\n".
    response_template = "<|im_start|>assistant\n"
    response_template_ids = tok.encode(response_template, add_special_tokens=False)
    _log("train", f"response_template_ids (len={len(response_template_ids)}): {response_template_ids}")
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tok,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
        processing_class=tok,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    _log("train", "Starting training...")
    train_result = trainer.train()
    _log("train", "Training complete.")

    # Save only the LoRA adapter
    adapter_dir = out_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))

    if trainer.is_world_process_zero():
        metrics = train_result.metrics
        metrics_path = out_dir / "train_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
        _log("train", f"Metrics: {metrics_path}")

        # Dump full log history for audit
        history_path = out_dir / "log_history.json"
        history_path.write_text(json.dumps(trainer.state.log_history, indent=2, default=str))
        _log("train", f"Log history: {history_path}")
        _log("train", f"Best val loss seen: {trainer.state.best_metric}")
        _log("train", f"Final adapter: {adapter_dir}")


# ---------------------------------------------------------------------------
# Section 7: Post-training evaluation + delta report
# ---------------------------------------------------------------------------


def cmd_eval() -> None:
    """Evaluate tuned model and produce delta-vs-baseline honesty report."""
    _ensure_dirs()
    _set_gpus()
    suffix = os.environ.get("DISTILL_OUTPUT_SUFFIX", "")
    adapter_path = PATHS["checkpoints"] / (CONFIG["output_name"] + suffix) / "final_adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(f"No adapter at {adapter_path}. Run `train` first.")
    baseline_path = PATHS["reports"] / "baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"No baseline at {baseline_path}. Run `baseline` BEFORE `train` for honest delta."
        )

    _log("eval", "Evaluating TUNED model...")
    tuned = _evaluate_model(model_kind="tuned", adapter_path=adapter_path)
    (PATHS["reports"] / "tuned.json").write_text(json.dumps(tuned, indent=2))
    _print_eval_summary("TUNED", tuned)

    baseline = json.loads(baseline_path.read_text())
    delta = {
        "test_split_loss_delta": tuned["test_split_loss"] - baseline["test_split_loss"],
        "test_split_ppl_delta": tuned["test_split_perplexity"] - baseline["test_split_perplexity"],
        "gsm8k_accuracy_delta": tuned["gsm8k"]["accuracy"] - baseline["gsm8k"]["accuracy"],
    }

    # Honesty flags
    flags = []
    if delta["test_split_loss_delta"] > 0:
        flags.append("WARN: test loss got WORSE after training.")
    if delta["gsm8k_accuracy_delta"] < -0.02:
        flags.append(
            "WARN: GSM8K accuracy dropped >2pp. Possible overfit to teacher style "
            "at the cost of generalization."
        )
    if delta["test_split_loss_delta"] < -1.0 and abs(delta["gsm8k_accuracy_delta"]) < 0.01:
        flags.append(
            "WARN: Huge test-loss drop with ~zero GSM8K change. Pattern consistent "
            "with memorization/reward-hacking; verify test set was frozen."
        )

    report = {
        "baseline": baseline,
        "tuned": tuned,
        "delta": delta,
        "flags": flags,
        "config": CONFIG,
        "notes": {
            "gpus_used": CUDA_VISIBLE,
            "gpus_total_requested": "3 (0,1,2)",
            "gpus_excluded": "GPU 1 (occupied by ollama per user direction)",
            "method": "LoRA r=32 SFT with completion-only loss masking",
            "eval_probes": ["held-out test split (frozen)", "GSM8K 100-sample (fixed seed)"],
        },
    }
    report_path = PATHS["reports"] / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    _log("eval", f"Final report: {report_path}")

    print("\n========== HONESTY REPORT ==========")
    print(f"  test_split_loss   base={baseline['test_split_loss']:.4f} tuned={tuned['test_split_loss']:.4f} Δ={delta['test_split_loss_delta']:+.4f}")
    print(f"  test_split_ppl    base={baseline['test_split_perplexity']:.2f}  tuned={tuned['test_split_perplexity']:.2f}  Δ={delta['test_split_ppl_delta']:+.2f}")
    print(f"  gsm8k_accuracy    base={baseline['gsm8k']['accuracy']:.3f} tuned={tuned['gsm8k']['accuracy']:.3f} Δ={delta['gsm8k_accuracy_delta']:+.3f}")
    if flags:
        print("  FLAGS:")
        for flag in flags:
            print(f"    - {flag}")
    else:
        print("  FLAGS: none (no obvious reward-hacking signatures)")
    print("====================================\n")


# ---------------------------------------------------------------------------
# Section 8: Export — merge LoRA, convert to GGUF, register with Ollama
# ---------------------------------------------------------------------------


def cmd_export() -> None:
    _ensure_dirs()
    _set_gpus()
    adapter_path = PATHS["checkpoints"] / CONFIG["output_name"] / "final_adapter"
    merged_dir = PATHS["merged"] / CONFIG["output_name"]
    gguf_dir = PATHS["gguf"] / CONFIG["output_name"]
    gguf_dir.mkdir(parents=True, exist_ok=True)

    _merge_lora(adapter_path, merged_dir)
    gguf_path = _convert_to_gguf(merged_dir, gguf_dir)
    quantized = _quantize_gguf(gguf_path, gguf_dir)
    _create_ollama_model(quantized)


def _merge_lora(adapter_path: Path, merged_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if merged_dir.exists() and any(merged_dir.iterdir()):
        _log("export", f"Merged dir already populated: {merged_dir} (skipping merge)")
        return
    merged_dir.mkdir(parents=True, exist_ok=True)
    _log("export", "Loading base model for merge...")
    base = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"], torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    _log("export", f"Attaching adapter {adapter_path}")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    _log("export", "Merging LoRA weights into base...")
    merged = model.merge_and_unload()
    _log("export", f"Saving merged model to {merged_dir}")
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True).save_pretrained(
        str(merged_dir)
    )


def _convert_to_gguf(merged_dir: Path, gguf_dir: Path) -> Path:
    """Clone llama.cpp if needed and run convert_hf_to_gguf.py."""
    llama = PATHS["llama_cpp"]
    if not llama.exists():
        llama.parent.mkdir(parents=True, exist_ok=True)
        _log("export", f"Cloning llama.cpp to {llama} ...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", str(llama)]
        )
    # Our training venv already has the deps convert_hf_to_gguf.py needs
    # (torch, transformers>=5, safetensors, gguf, sentencepiece, protobuf).
    # We deliberately do NOT run `pip install -r requirements-convert_hf_to_gguf.txt`
    # because upstream pins `transformers<5.0.0` which would downgrade our
    # transformers and break Qwen3.5 model loading. If convert_hf_to_gguf
    # complains about a missing lib, install only that specific package.
    vpy = VENV / "bin" / "python"

    conv = llama / "convert_hf_to_gguf.py"
    vpy = VENV / "bin" / "python"
    out_f16 = gguf_dir / f"{CONFIG['output_name']}.f16.gguf"
    if out_f16.exists():
        _log("export", f"F16 GGUF already present: {out_f16}")
        return out_f16
    _log("export", f"Converting merged HF model → {out_f16}")
    subprocess.check_call(
        [
            str(vpy),
            str(conv),
            str(merged_dir),
            "--outfile",
            str(out_f16),
            "--outtype",
            "f16",
        ]
    )
    return out_f16


def _quantize_gguf(f16_path: Path, gguf_dir: Path) -> dict:
    """Build llama-quantize if needed and produce Q4_K_M + Q8_0 variants."""
    llama = PATHS["llama_cpp"]
    build = llama / "build"
    quantize_bin = build / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        _log("export", "Building llama.cpp quantize binary (cmake -B build)...")
        subprocess.check_call(["cmake", "-B", str(build), "-S", str(llama), "-DLLAMA_CURL=OFF"])
        subprocess.check_call(["cmake", "--build", str(build), "--target", "llama-quantize", "-j"])

    results = {"f16": str(f16_path)}
    for qname, label in [("Q4_K_M", "q4km"), ("Q8_0", "q80")]:
        out = gguf_dir / f"{CONFIG['output_name']}.{label}.gguf"
        if out.exists():
            _log("export", f"{qname} GGUF already present: {out}")
        else:
            _log("export", f"Quantizing → {qname}: {out}")
            subprocess.check_call([str(quantize_bin), str(f16_path), str(out), qname])
        results[label] = str(out)
    return results


def _create_ollama_model(quantized: dict) -> None:
    """Create an Ollama model from the Q4_K_M GGUF.

    We deliberately do NOT set a TEMPLATE block — the GGUF already carries
    Qwen3.5's Jinja chat template (with full tool-calling / function-calling
    support). Overriding the template with a simplified one strips the
    tool-call path and makes Ollama report "does not support tools".
    """
    if shutil.which("ollama") is None:
        _log("export", "ollama CLI not found on PATH; skipping model registration.")
        return
    gguf_dir = PATHS["gguf"] / CONFIG["output_name"]
    modelfile = gguf_dir / "Modelfile"
    q4_path = quantized["q4km"]
    # CRITICAL: RENDERER + PARSER must be set to qwen3.5 for Ollama to emit
    # structured message.tool_calls. Without them `ollama show` reports
    # "no tools capability" and /api/chat ignores the tools field.
    # See tool-calling preservation gate docs in this file for context.
    rel_q4 = Path(q4_path).name  # relative; Modelfile lives next to the GGUF
    modelfile.write_text(
        f"""# Modelfile for Ollama — {CONFIG['output_name']}
#
# RENDERER + PARSER directives are REQUIRED for Qwen3.5 tool calling.
# They wire Ollama's native Go handlers so the model's <tool_call> format
# gets translated into structured message.tool_calls in /api/chat responses.
#
# Swap the FROM line to use a different quant:
#   ./{CONFIG['output_name']}.f16.gguf   (~17.9 GB)
#   ./{CONFIG['output_name']}.q80.gguf   (~9.5 GB)
#   ./{CONFIG['output_name']}.q4km.gguf  (~5.6 GB) <- default

FROM ./{rel_q4}

RENDERER qwen3.5
PARSER qwen3.5

PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER stop "<|im_end|>"
"""
    )
    name = "qwen3.5-9b-qwen3.6-distilled:q4km"
    _log("export", f"Creating ollama model: {name}")
    try:
        subprocess.check_call(["ollama", "create", name, "-f", str(modelfile)])
        _log("export", f"OK — try:  ollama run {name}")
    except subprocess.CalledProcessError as e:
        _log("export", f"ollama create failed: {e}")


# ---------------------------------------------------------------------------
# Section 8.5: Tool-calling preservation gate
# ---------------------------------------------------------------------------
#
# The gate enforces the Ollama contract: tools in request, structured
# tool_calls in response, tool-role messages for results, correct follow-up.
#
# Four buckets (all indexed in a SHA256-locked manifest):
#   single   — exactly one tool call required
#   parallel — two tool calls in one turn
#   none     — plain question, no tool call
#   post     — after we feed back a tool result, final answer must use it
#
# Seven metrics, all reported base-vs-tuned with regression flags:
#   tool_calls_present_rate
#   tool_selection_accuracy
#   argument_schema_valid
#   argument_accuracy
#   parallel_tool_call_accuracy
#   no_call_when_unneeded
#   post_tool_response_accuracy


_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a numeric math expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression, e.g. '234*567+89'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia and return a short snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time in a named timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone, e.g. 'Europe/London'"}
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string", "description": "Source ISO code, e.g. 'USD'"},
                    "to_currency": {"type": "string", "description": "Target ISO code, e.g. 'EUR'"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
]


_TOOL_BENCH_ROWS = [
    # ---- single-tool bucket (6) ----
    {"id": "s1", "bucket": "single", "user": "What's the current weather in Paris?",
     "expected": {"tool": "get_weather", "args_match": {"city": "Paris"}}},
    {"id": "s2", "bucket": "single", "user": "Can you tell me the temperature in Tokyo in fahrenheit?",
     "expected": {"tool": "get_weather", "args_match": {"city": "Tokyo", "unit": "fahrenheit"}}},
    {"id": "s3", "bucket": "single", "user": "Compute 234 times 567 plus 89 for me.",
     "expected": {"tool": "calculator", "args_match": {"expression_contains": ["234", "567", "89"]}}},
    {"id": "s4", "bucket": "single", "user": "Look up information about the Eiffel Tower on Wikipedia.",
     "expected": {"tool": "search_wikipedia", "args_match": {"query_contains": ["Eiffel", "Tower"]}}},
    {"id": "s5", "bucket": "single", "user": "What time is it right now in London?",
     "expected": {"tool": "get_current_time", "args_match": {"timezone_contains": ["London"]}}},
    {"id": "s6", "bucket": "single", "user": "Convert 100 US dollars to euros.",
     "expected": {"tool": "convert_currency",
                  "args_match": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}}},

    # ---- parallel-tool bucket (4) ----
    {"id": "p1", "bucket": "parallel",
     "user": "I need the current weather in both Paris and Tokyo. Please get both.",
     "expected": {"tools": ["get_weather", "get_weather"],
                  "args_any": [{"city": "Paris"}, {"city": "Tokyo"}]}},
    {"id": "p2", "bucket": "parallel",
     "user": "Please convert 100 USD to EUR and also convert 100 USD to JPY.",
     "expected": {"tools": ["convert_currency", "convert_currency"],
                  "args_any": [
                      {"amount": 100, "from_currency": "USD", "to_currency": "EUR"},
                      {"amount": 100, "from_currency": "USD", "to_currency": "JPY"},
                  ]}},
    {"id": "p3", "bucket": "parallel",
     "user": "What's the time in London AND New York right now? Use the tool twice.",
     "expected": {"tools": ["get_current_time", "get_current_time"],
                  "args_any": [
                      {"timezone_contains": ["London"]},
                      {"timezone_contains": ["New_York", "New York", "America"]},
                  ]}},
    {"id": "p4", "bucket": "parallel",
     "user": "Compute 15*23 with the calculator and also search Wikipedia for 'quantum computing'.",
     "expected": {"tools": ["calculator", "search_wikipedia"],
                  "args_any": [
                      {"expression_contains": ["15", "23"]},
                      {"query_contains": ["quantum", "computing"]},
                  ]}},

    # ---- none bucket (5) ----
    {"id": "n1", "bucket": "none", "user": "What is the capital of France?"},
    {"id": "n2", "bucket": "none", "user": "Write a one-line haiku about autumn."},
    {"id": "n3", "bucket": "none", "user": "In one sentence, explain what recursion is."},
    {"id": "n4", "bucket": "none", "user": "How do you say hello in Spanish?"},
    {"id": "n5", "bucket": "none", "user": "In what year did World War II end?"},

    # ---- post-tool-response bucket (5) ----
    {"id": "d1", "bucket": "post",
     "user": "What's the current weather in Berlin?",
     "expected_first_tool": "get_weather",
     "mock_tool_result": "18 degrees celsius, rainy",
     "final_contains_any": ["18", "rain"]},
    {"id": "d2", "bucket": "post",
     "user": "Compute 1234 * 5678 for me.",
     "expected_first_tool": "calculator",
     "mock_tool_result": "7006652",
     "final_contains_any": ["7006652", "7,006,652"]},
    {"id": "d3", "bucket": "post",
     "user": "I'd like to convert 50 US dollars to euros. What's the result?",
     "expected_first_tool": "convert_currency",
     "mock_tool_result": "45.50 EUR",
     "final_contains_any": ["45.5", "45.50", "euro"]},
    {"id": "d4", "bucket": "post",
     "user": "Search Wikipedia for 'Python programming language'. Then summarize.",
     "expected_first_tool": "search_wikipedia",
     "mock_tool_result": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.",
     "final_contains_any": ["Guido", "1991", "high-level", "interpreted"]},
    {"id": "d5", "bucket": "post",
     "user": "What time is it right now in Dubai (Asia/Dubai)?",
     "expected_first_tool": "get_current_time",
     "mock_tool_result": "16:30",
     "final_contains_any": ["16:30", "4:30"]},
]


def cmd_prepare_tools() -> None:
    """Write the tool-calling benchmark + manifest to disk (locked/frozen)."""
    _ensure_dirs()
    d = PATHS["tool_bench_dir"]
    d.mkdir(parents=True, exist_ok=True)
    PATHS["tool_defs"].write_text(json.dumps(_TOOL_DEFS, indent=2))
    with PATHS["tool_bench"].open("w") as f:
        for row in _TOOL_BENCH_ROWS:
            f.write(json.dumps(row) + "\n")
    manifest = {
        "n_rows": len(_TOOL_BENCH_ROWS),
        "buckets": {
            b: sum(1 for r in _TOOL_BENCH_ROWS if r["bucket"] == b)
            for b in ["single", "parallel", "none", "post"]
        },
        "sha256_tools": _file_sha(PATHS["tool_defs"]),
        "sha256_bench": _file_sha(PATHS["tool_bench"]),
        "ollama_url": OLLAMA_URL,
        "base_model_name": OLLAMA_BASE_MODEL,
        "tuned_model_name": OLLAMA_TUNED_MODEL,
    }
    PATHS["tool_manifest"].write_text(json.dumps(manifest, indent=2))
    _log("tools", f"Wrote {len(_TOOL_BENCH_ROWS)} rows to {PATHS['tool_bench']}")
    _log("tools", f"Buckets: {manifest['buckets']}")
    _log("tools", f"Manifest: {PATHS['tool_manifest']}")


# --- validation helpers --------------------------------------------------


def _json_type_of(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "unknown"


def _validate_args_schema(args: dict, schema: dict) -> tuple[bool, str]:
    """Tiny JSON-schema checker: required, type, enum."""
    if not isinstance(args, dict):
        return False, f"args is {type(args).__name__}, not object"
    props = schema.get("properties", {})
    required = schema.get("required", [])
    for key in required:
        if key not in args:
            return False, f"missing required field {key!r}"
    for key, val in args.items():
        if key not in props:
            # Extra fields: allowed by default
            continue
        expected_type = props[key].get("type")
        actual_type = _json_type_of(val)
        if expected_type == "number" and actual_type == "integer":
            pass  # integers are valid numbers
        elif expected_type and actual_type != expected_type:
            return False, f"{key}: expected {expected_type}, got {actual_type}"
        enum = props[key].get("enum")
        if enum is not None and val not in enum:
            return False, f"{key}: {val!r} not in enum {enum}"
    return True, ""


def _args_accuracy(actual: dict, expected: dict) -> bool:
    """Fuzzy accuracy: supports scalar match and *_contains substring checks."""
    if not isinstance(actual, dict):
        return False
    for key, want in expected.items():
        if key.endswith("_contains"):
            target_key = key.rsplit("_contains", 1)[0]
            # Accept any actual key whose value contains every needle (case-insensitive)
            candidate = actual.get(target_key, "")
            candstr = str(candidate).lower()
            if not all(str(w).lower() in candstr for w in want):
                return False
        else:
            if key not in actual:
                return False
            a = actual[key]
            # Case-insensitive string compare; numeric exact
            if isinstance(want, str):
                if str(a).strip().lower() != want.strip().lower():
                    return False
            else:
                try:
                    if float(a) != float(want):
                        return False
                except (TypeError, ValueError):
                    return False
    return True


def _ollama_chat(model: str, messages: list, tools: list | None = None, timeout: float = 300.0):
    import httpx
    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _run_tools_benchmark(model_name: str, label: str) -> dict:
    """Run the full tool-calling benchmark against an Ollama model.

    Returns dict with per-row results + aggregate metrics.
    """
    import httpx

    tools = json.loads(PATHS["tool_defs"].read_text())
    rows = [json.loads(l) for l in PATHS["tool_bench"].read_text().splitlines() if l.strip()]
    tool_schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}

    per_row = []
    # counters for aggregate metrics
    c_single_parallel_post_total = 0
    c_tool_present = 0
    c_tool_select_ok = 0
    c_tool_select_total = 0  # single + parallel (post handled separately)
    c_schema_ok = 0
    c_schema_total = 0
    c_arg_acc_ok = 0
    c_arg_acc_total = 0
    c_parallel_ok = 0
    c_parallel_total = 0
    c_none_ok = 0
    c_none_total = 0
    c_post_ok = 0
    c_post_total = 0

    for row in rows:
        rid = row["id"]
        bucket = row["bucket"]
        user_msg = {"role": "user", "content": row["user"]}
        _log("tools-eval", f"[{label}] {rid} ({bucket}): {row['user'][:60]}")
        row_result = {"id": rid, "bucket": bucket}
        try:
            resp = _ollama_chat(model_name, [user_msg], tools=tools)
        except httpx.HTTPStatusError as e:
            row_result["error"] = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            per_row.append(row_result)
            continue
        except Exception as e:
            row_result["error"] = f"{type(e).__name__}: {e}"
            per_row.append(row_result)
            continue

        msg = resp.get("message", {})
        tool_calls = msg.get("tool_calls") or []
        row_result["tool_calls_count"] = len(tool_calls)
        row_result["raw_tool_calls"] = tool_calls
        row_result["content"] = msg.get("content", "")[:500]

        # ---- metrics per bucket ----
        if bucket == "single":
            c_single_parallel_post_total += 1
            if tool_calls:
                c_tool_present += 1
            c_tool_select_total += 1
            exp = row["expected"]
            if tool_calls and tool_calls[0]["function"]["name"] == exp["tool"]:
                c_tool_select_ok += 1
                row_result["selected_ok"] = True
            else:
                row_result["selected_ok"] = False
            # schema valid + arg accuracy (only when there IS a call)
            if tool_calls:
                tc = tool_calls[0]
                fname = tc["function"]["name"]
                fargs = tc["function"].get("arguments", {})
                if fname in tool_schemas:
                    c_schema_total += 1
                    ok, why = _validate_args_schema(fargs, tool_schemas[fname])
                    if ok:
                        c_schema_ok += 1
                    row_result["schema_ok"] = ok
                    row_result["schema_why"] = why
                c_arg_acc_total += 1
                ok_args = (
                    tc["function"]["name"] == exp["tool"]
                    and _args_accuracy(fargs, exp["args_match"])
                )
                if ok_args:
                    c_arg_acc_ok += 1
                row_result["args_ok"] = ok_args

        elif bucket == "parallel":
            c_single_parallel_post_total += 1
            if tool_calls:
                c_tool_present += 1
            c_parallel_total += 1
            exp = row["expected"]
            want_tools = sorted(exp["tools"])
            got_tools = sorted([tc["function"]["name"] for tc in tool_calls])
            tools_match = want_tools == got_tools
            # Argument accuracy: each expected arg-set must match at least one actual call
            got_args = [tc["function"].get("arguments", {}) for tc in tool_calls]
            args_match = True
            for wanted in exp["args_any"]:
                if not any(_args_accuracy(got, wanted) for got in got_args):
                    args_match = False
                    break
            if tools_match and args_match:
                c_parallel_ok += 1
            row_result["parallel_tools_match"] = tools_match
            row_result["parallel_args_match"] = args_match
            # Schema valid across all parallel calls
            for tc in tool_calls:
                fname = tc["function"]["name"]
                fargs = tc["function"].get("arguments", {})
                if fname in tool_schemas:
                    c_schema_total += 1
                    ok, _ = _validate_args_schema(fargs, tool_schemas[fname])
                    if ok:
                        c_schema_ok += 1

        elif bucket == "none":
            c_none_total += 1
            if not tool_calls:
                c_none_ok += 1
            row_result["no_call_ok"] = not tool_calls

        elif bucket == "post":
            c_single_parallel_post_total += 1
            if tool_calls:
                c_tool_present += 1
            c_post_total += 1
            expected_first = row.get("expected_first_tool")
            first_ok = bool(tool_calls) and tool_calls[0]["function"]["name"] == expected_first
            row_result["first_tool_ok"] = first_ok
            if not first_ok:
                # Post-tool response accuracy requires a valid first call
                per_row.append(row_result)
                continue
            # Feed mock tool result back as role=tool message and continue chat
            # Ollama expects:  {"role": "tool", "content": "...", "tool_name": "..."}
            tool_reply = {
                "role": "tool",
                "content": row["mock_tool_result"],
                "tool_name": expected_first,
            }
            continued_messages = [user_msg, msg, tool_reply]
            try:
                resp2 = _ollama_chat(model_name, continued_messages, tools=tools)
                final = resp2.get("message", {}).get("content", "")
                row_result["post_final"] = final[:500]
                haystack = final.lower()
                needles = row["final_contains_any"]
                post_ok = any(str(n).lower() in haystack for n in needles)
                if post_ok:
                    c_post_ok += 1
                row_result["post_ok"] = post_ok
            except Exception as e:
                row_result["post_error"] = f"{type(e).__name__}: {e}"

        per_row.append(row_result)

    def _safe_div(a, b):
        return (a / b) if b else 0.0

    metrics = {
        "tool_calls_present_rate": _safe_div(c_tool_present, c_single_parallel_post_total),
        "tool_selection_accuracy": _safe_div(c_tool_select_ok, c_tool_select_total),
        "argument_schema_valid": _safe_div(c_schema_ok, c_schema_total),
        "argument_accuracy": _safe_div(c_arg_acc_ok, c_arg_acc_total),
        "parallel_tool_call_accuracy": _safe_div(c_parallel_ok, c_parallel_total),
        "no_call_when_unneeded": _safe_div(c_none_ok, c_none_total),
        "post_tool_response_accuracy": _safe_div(c_post_ok, c_post_total),
    }
    return {
        "label": label,
        "model": model_name,
        "metrics": metrics,
        "counts": {
            "single_parallel_post_total": c_single_parallel_post_total,
            "tool_present": c_tool_present,
            "tool_select_ok": c_tool_select_ok,
            "tool_select_total": c_tool_select_total,
            "schema_ok": c_schema_ok,
            "schema_total": c_schema_total,
            "arg_acc_ok": c_arg_acc_ok,
            "arg_acc_total": c_arg_acc_total,
            "parallel_ok": c_parallel_ok,
            "parallel_total": c_parallel_total,
            "none_ok": c_none_ok,
            "none_total": c_none_total,
            "post_ok": c_post_ok,
            "post_total": c_post_total,
        },
        "per_row": per_row,
    }


def _ollama_model_has_tools(model_name: str) -> bool:
    """Return True iff `ollama show` lists 'tools' in Capabilities."""
    try:
        out = subprocess.run(
            ["ollama", "show", model_name], capture_output=True, text=True, timeout=30
        )
        return "tools" in out.stdout.lower()
    except Exception:
        return False


def cmd_baseline_tools() -> None:
    """Run the tool benchmark against the UNTOUCHED base model via Ollama."""
    _ensure_dirs()
    if not PATHS["tool_manifest"].exists():
        raise FileNotFoundError("Run `prepare-tools` first.")
    if not _ollama_model_has_tools(OLLAMA_BASE_MODEL):
        raise RuntimeError(
            f"Base model {OLLAMA_BASE_MODEL} in Ollama does not advertise `tools` capability. "
            "Check `ollama show` output."
        )
    _log("tools-baseline", f"Running benchmark against {OLLAMA_BASE_MODEL}")
    result = _run_tools_benchmark(OLLAMA_BASE_MODEL, "base")
    out = PATHS["reports"] / "tool_baseline.json"
    out.write_text(json.dumps(result, indent=2))
    _log("tools-baseline", f"Saved {out}")
    _print_tool_metrics("BASE (Ollama)", result["metrics"])


def cmd_eval_tools() -> None:
    """Run the tool benchmark against the TUNED GGUF via Ollama."""
    _ensure_dirs()
    if not PATHS["tool_manifest"].exists():
        raise FileNotFoundError("Run `prepare-tools` first.")
    if not _ollama_model_has_tools(OLLAMA_TUNED_MODEL):
        raise RuntimeError(
            f"Tuned model {OLLAMA_TUNED_MODEL} in Ollama does not advertise `tools` capability. "
            "Did you (re)run `export`? RENDERER/PARSER must be set to qwen3.5 in the Modelfile."
        )
    _log("tools-eval", f"Running benchmark against {OLLAMA_TUNED_MODEL}")
    result = _run_tools_benchmark(OLLAMA_TUNED_MODEL, "tuned")
    out = PATHS["reports"] / "tool_tuned.json"
    out.write_text(json.dumps(result, indent=2))
    _log("tools-eval", f"Saved {out}")
    _print_tool_metrics("TUNED (Ollama)", result["metrics"])


def _print_tool_metrics(tag: str, m: dict) -> None:
    print(f"\n===== {tag} tool-call metrics =====")
    for k in [
        "tool_calls_present_rate",
        "tool_selection_accuracy",
        "argument_schema_valid",
        "argument_accuracy",
        "parallel_tool_call_accuracy",
        "no_call_when_unneeded",
        "post_tool_response_accuracy",
    ]:
        print(f"  {k:32s} = {m[k]:.3f}")
    print("=======================================\n")


def cmd_verify_ollama() -> None:
    """End-to-end Ollama gate: compare base vs tuned, compute deltas, flag regressions,
    merge into final_report.json, and print the gate verdict.
    """
    _ensure_dirs()
    base_path = PATHS["reports"] / "tool_baseline.json"
    tuned_path = PATHS["reports"] / "tool_tuned.json"
    if not base_path.exists() or not tuned_path.exists():
        raise FileNotFoundError(
            "Need tool_baseline.json and tool_tuned.json. Run prepare-tools, baseline-tools, eval-tools first."
        )
    base = json.loads(base_path.read_text())
    tuned = json.loads(tuned_path.read_text())

    # Verify the tool-call contract itself landed (not just text content)
    contract_ok = any(r.get("tool_calls_count", 0) > 0 for r in tuned["per_row"] if r["bucket"] in ("single", "parallel", "post"))

    deltas = {}
    regress_flags = []
    # Regression rule: any metric where tuned < base by more than epsilon is a regression.
    # epsilon = 0.05 (5pp) — strict gate.
    epsilon = 0.05
    for k, bv in base["metrics"].items():
        tv = tuned["metrics"][k]
        deltas[k] = tv - bv
        if tv + epsilon < bv:
            regress_flags.append(
                f"REGRESSION: {k} dropped by {bv - tv:+.3f} (base={bv:.3f} tuned={tv:.3f}, epsilon={epsilon})"
            )

    # Merge into final_report.json
    final_path = PATHS["reports"] / "final_report.json"
    if final_path.exists():
        final_report = json.loads(final_path.read_text())
    else:
        final_report = {}

    gate_passed = contract_ok and not regress_flags
    final_report["tool_calling"] = {
        "ran": True,
        "contract_ok": contract_ok,
        "base_metrics": base["metrics"],
        "tuned_metrics": tuned["metrics"],
        "deltas": deltas,
        "regression_flags": regress_flags,
        "gate_passed": gate_passed,
        "base_counts": base["counts"],
        "tuned_counts": tuned["counts"],
        "base_model": base["model"],
        "tuned_model": tuned["model"],
    }
    final_path.write_text(json.dumps(final_report, indent=2))
    _log("tools-verify", f"Merged tool gate results into {final_path}")

    # Print the verdict
    print("\n========== TOOL-CALLING PRESERVATION GATE ==========")
    print(f"Contract ok (structured tool_calls returned): {contract_ok}")
    print(f"Base model : {base['model']}")
    print(f"Tuned model: {tuned['model']}")
    print()
    print(f"{'metric':34s}  {'base':>7s}  {'tuned':>7s}  {'delta':>7s}")
    for k, bv in base["metrics"].items():
        tv = tuned["metrics"][k]
        dv = deltas[k]
        marker = "  REGRESS" if tv + epsilon < bv else ""
        print(f"{k:34s}  {bv:7.3f}  {tv:7.3f}  {dv:+7.3f}{marker}")
    print()
    if regress_flags:
        print("FLAGS:")
        for f in regress_flags:
            print(f"  - {f}")
    else:
        print("FLAGS: none")
    print()
    print(f"GATE {'PASSED ✓' if gate_passed else 'FAILED ✗'}")
    print("====================================================\n")

    return final_report


# ---------------------------------------------------------------------------
# Section 9: Model card + HF Hub upload
# ---------------------------------------------------------------------------


def _generate_model_card(report: dict) -> str:
    """Build an honest, numbers-faithful model card from the final report.

    The card is strictly conditional on what was measured:
      * reasoning metrics come from final_report.json
      * tool-calling wording depends on report["tool_calling"]:
          - absent        -> "inherited, unverified" section, NO tool tags
          - ran + passed  -> "measured preservation" section WITH tool tags
          - ran + failed  -> explicit regression block, NO tool tags, release blocked
    """
    b = report["baseline"]
    t = report["tuned"]
    d = report["delta"]
    flags = report.get("flags", [])
    cfg = report["config"]
    notes = report["notes"]
    tool = report.get("tool_calling")  # may be None
    hard = report.get("hard_tool_tests")  # may be None
    img = report.get("image_handling")  # may be None

    flag_block = ""
    if flags:
        flag_lines = "\n".join(f"- {f}" for f in flags)
        flag_block = f"\n### Honesty flags raised\n\n{flag_lines}\n"

    # ---- tool-calling tags: ONLY if we measured preservation end-to-end
    base_tags = [
        "reasoning", "distillation", "lora", "qwen3.5", "sft",
    ]
    tool_tags = []
    if tool and tool.get("ran") and tool.get("gate_passed"):
        tool_tags = ["tool-use", "function-calling"]
    tags_yaml = "\n".join(f"- {t}" for t in base_tags + tool_tags)

    # ---- hard test section (optional, appended below the primary tool section)
    hard_section = ""
    if hard and hard.get("ran"):
        hard_summary = hard["summary"]
        hard_base_key = next((k for k in hard_summary if "distilled" not in k), None)
        hard_tuned_key = next((k for k in hard_summary if "distilled" in k), None)
        if hard_base_key and hard_tuned_key:
            hb = hard_summary[hard_base_key]
            ht = hard_summary[hard_tuned_key]
            all_rows_base = {row["id"]: row for row in hb["rows"]}
            all_rows_tuned = {row["id"]: row for row in ht["rows"]}
            hard_ids = sorted(
                all_rows_base.keys(),
                key=lambda x: (x[0], int("".join(c for c in x[1:] if c.isdigit()) or 0)),
            )
            rows_md = []
            for rid in hard_ids:
                rb = all_rows_base.get(rid, {})
                rt = all_rows_tuned.get(rid, {})
                row_label = (rb.get("label") or rt.get("label") or "").replace("|", "\\|")
                base_mark = "✅" if rb.get("pass") else "❌"
                tuned_mark = "✅" if rt.get("pass") else "❌"
                rows_md.append(f"| {rid} | {row_label[:70]} | {base_mark} | {tuned_mark} |")
            dec = hard.get("decoding", {})
            hard_notes = hard.get("notes", {})
            notes_md = "\n".join(f"- **{k}**: {v}" for k, v in hard_notes.items())
            hard_section = f"""
### Stress test — 22 hard tool-calling scenarios (deterministic)

Decoding: `temperature={dec.get('temperature', 0)}`, `top_p={dec.get('top_p', 1)}`, `top_k={dec.get('top_k', 1)}`, `seed={dec.get('seed', 42)}`.
Endpoint: {hard.get('endpoint', 'Ollama chat API')}.

Scores: **{hb['passed']} / {hb['total']}** base, **{ht['passed']} / {ht['total']}** tuned.

| # | Test | Base | Tuned |
|---|---|:---:|:---:|
{chr(10).join(rows_md)}

**Honest notes on the failures:**

{notes_md}
"""

    # ---- image-handling section
    image_section = ""
    if img and img.get("ran"):
        models = img.get("models", {})
        err_msg = img.get("tuned_error_message", "")
        root_cause = img.get("notes", {}).get("root_cause", "")
        remedy = img.get("notes", {}).get("remedy", "")
        tuned_key = next((k for k in models if "distilled" in k), None)
        base_key = next((k for k in models if "distilled" not in k), None)
        tuned_statuses = models.get(tuned_key, {}).get("all_status", []) if tuned_key else []
        base_passed = models.get(base_key, {}).get("passed", "?") if base_key else "?"
        base_total = models.get(base_key, {}).get("total", "?") if base_key else "?"
        image_section = f"""
## Image handling — NOT SUPPORTED

This distilled model does **not** support image inputs. Attempting to pass
`images` to Ollama's `/api/chat` returns **HTTP 500**:

```
{err_msg}
```

### Why

{root_cause}

### Verified with a small probe (3 rendered test images)

| Model | Image-probe score | HTTP statuses |
|---|:---:|:---:|
| `{base_key or 'qwen3.5:9b'}` (base) | {base_passed} / {base_total} | 200 (serves images) |
| `{tuned_key or 'qwen3.5-9b-qwen3.6-distilled:q4km'}` (tuned) | 0 / 3 | {tuned_statuses} (no vision) |

The base model *serves* image requests (Ollama reports `vision` in its
Capabilities) but its text-in-image OCR on the three probes was weak
(e.g. "HELLO" → "HELO", "42" → "44", "BANANA" → "barna"). Do not assume
the base model is a reliable OCR tool just because image requests return
HTTP 200.

### Remedy

{remedy}
"""

    # ---- tool-calling section body
    if tool is None or not tool.get("ran"):
        tool_section = """## Tool calling

Tool calling is **inherited from the base model family**. This fine-tune did not
include tool-use training data, and tool-calling behavior was **not separately
benchmarked** after distillation. Do not assume tool-calling quality is
preserved without checking the release benchmark (`app.py prepare-tools →
baseline-tools → eval-tools → verify-ollama`).
"""
    elif tool.get("gate_passed"):
        bm = tool["base_metrics"]
        tm = tool["tuned_metrics"]
        dlt = tool["deltas"]
        # Build side-by-side table
        rows = []
        for k in [
            "tool_calls_present_rate",
            "tool_selection_accuracy",
            "argument_schema_valid",
            "argument_accuracy",
            "parallel_tool_call_accuracy",
            "no_call_when_unneeded",
            "post_tool_response_accuracy",
        ]:
            rows.append(f"| {k} | {bm[k]:.3f} | {tm[k]:.3f} | {dlt[k]:+.3f} |")
        metrics_table = "\n".join(rows)
        tool_section = f"""## Tool calling

This model was evaluated for tool calling through the **Ollama chat API**.
Tool-use preservation was checked on a locked held-out benchmark covering
single-tool, parallel-tool, no-tool, and post-tool-response turns.

The serving contract expected by downstream runtimes is:

- assistant emits **structured** `tool_calls` (not free-text `<tool_call>` blocks)
- tool results are returned as `tool`-role messages
- the model continues using those results correctly

### Side-by-side metrics (base vs tuned, same locked benchmark)

Base: `{tool['base_model']}`
Tuned: `{tool['tuned_model']}`

| Metric | Base | Tuned | Δ |
|---|---:|---:|---:|
{metrics_table}

Regression gate: **PASSED** (no metric regressed beyond epsilon=0.05).
Full per-row results, including the exact `tool_calls` returned by Ollama, are
in `final_report.json → tool_calling`.
"""
    else:
        # ran but failed
        flag_lines = "\n".join(f"  - {f}" for f in tool.get("regression_flags", []))
        tool_section = f"""## Tool calling — RELEASE BLOCKED

This checkpoint was benchmarked for tool calling through the Ollama chat API
and **regressed** relative to the base model. Do NOT rely on tool calling with
this adapter.

Contract ok (structured `tool_calls` returned): {tool.get('contract_ok')}

### Regression flags
{flag_lines or '  - (none logged, but gate_passed=False)'}

Full per-row results are in `final_report.json → tool_calling`.
"""

    # Build the card — Jinja-free, plain f-strings
    return f"""---
base_model: {cfg['base_model']}
datasets:
- {cfg['dataset_id']}
library_name: peft
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
{tags_yaml}
---

# {cfg['output_name']}

LoRA adapter distilled from the reasoning traces in
[`{cfg['dataset_id']}`](https://huggingface.co/datasets/{cfg['dataset_id']})
on top of the base model
[`{cfg['base_model']}`](https://huggingface.co/{cfg['base_model']}).

Teacher model: **Qwen3.6-plus** (as declared by the source dataset).
Student model: **{cfg['base_model']}** (instruct, text subnet via `Qwen3_5ForCausalLM`).

## Training data

- **Source:** `{cfg['dataset_id']}` (500 examples, single user/assistant chat pairs where the assistant contains explicit `<think>...</think>` reasoning blocks).
- **Domain coverage:** coding, mathematics, finance, medicine, economics.
- **Split:** deterministic (seed `{cfg['seed']}`), 85 / 7.5 / 7.5 = **425 train / 37 val / 38 test**.
- **Locked test set** — SHA256 recorded in `data/splits/manifest.json` and never touched by training.

## Training recipe

- **Method:** LoRA SFT with completion-only loss masking (loss computed on assistant tokens only).
- **LoRA:** `r={cfg['lora']['r']}`, `alpha={cfg['lora']['alpha']}`, `dropout={cfg['lora']['dropout']}`, `target_modules={cfg['lora']['target_modules']}` (excludes embeddings and `lm_head` per Qwen's PEFT recommendation).
- **Optimizer:** AdamW, `lr={cfg['train']['learning_rate']}`, cosine schedule, warmup `{cfg['train']['warmup_ratio']}`, weight decay `{cfg['train']['weight_decay']}`.
- **Batching:** per-device batch `{cfg['train']['per_device_batch_size']}`, grad accum `{cfg['train']['grad_accum_steps']}`, epochs `{cfg['train']['num_epochs']}`, max sequence length `{cfg['train']['max_seq_length']}`.
- **Precision:** `bf16` with gradient checkpointing.
- **Early stopping:** patience `{cfg['train']['early_stopping_patience']}` on validation loss.
- **Hardware:** {notes['gpus_used']} — {notes.get('gpus_total_requested', '3 A100s')}.

## Evaluation (anti-reward-hacking)

All eval runs use fixed seeds and a test split that was frozen **before** training began.

### Probe 1 — held-out test split (teacher-forced loss on assistant tokens)

| Metric                 | Base     | Tuned    | Δ           |
|------------------------|----------|----------|-------------|
| Test loss              | {b['test_split_loss']:.4f} | {t['test_split_loss']:.4f} | **{d['test_split_loss_delta']:+.4f}** |
| Test perplexity        | {b['test_split_perplexity']:.2f}   | {t['test_split_perplexity']:.2f}   | **{d['test_split_ppl_delta']:+.2f}**  |

### Probe 2 — GSM8K (100-sample greedy, external honesty probe)

| Metric         | Base | Tuned | Δ |
|----------------|------|-------|---|
| GSM8K accuracy | {b['gsm8k']['accuracy']:.3f} | {t['gsm8k']['accuracy']:.3f} | **{d['gsm8k_accuracy_delta']:+.3f}** |

GSM8K samples are drawn from `openai/gsm8k` (test split) with a fixed seed
(`{cfg['eval']['gsm8k_seed']}`). The **same** 100 questions are used for base
and tuned so the delta is directly comparable.
{flag_block}
{hard_section}
{image_section}

## Running this model with Ollama

**Recommended path — download the Modelfile and build locally.**
Direct `ollama pull hf.co/...` will auto-derive a Modelfile from GGUF
metadata and may lose tool support. Using the Modelfile in this repo is
the reliable way to preserve `tools` + `thinking` capabilities:

```bash
# Install hf CLI if needed (part of huggingface_hub):  pip install -U huggingface_hub
hf download cudabenchmarktest/qwen3.5-9b-qwen3.6-reasoning-distilled-GGUF \
    Modelfile qwen3.5-9b-qwen3.6-reasoning-distilled.q4km.gguf \
    --local-dir ./qwen-distilled

cd qwen-distilled
ollama create qwen3.5-9b-qwen3.6-distilled:q4km -f Modelfile
ollama show    qwen3.5-9b-qwen3.6-distilled:q4km
#   Capabilities should include: tools, thinking

# Tool-calling smoke test
curl -s http://localhost:11434/api/chat -d '{{
  "model": "qwen3.5-9b-qwen3.6-distilled:q4km",
  "messages": [{{"role": "user", "content": "What is the weather in Paris?"}}],
  "tools": [{{"type":"function","function":{{"name":"get_weather","description":"Get current weather","parameters":{{"type":"object","properties":{{"city":{{"type":"string"}}}},"required":["city"]}}}}}}],
  "stream": false
}}' | jq .message.tool_calls
```

To use a different quant, download the matching `.gguf` and edit the
Modelfile's `FROM` line (or use one of the comment-suggested filenames):

- `qwen3.5-9b-qwen3.6-reasoning-distilled.q4km.gguf` (~5.6 GB) — recommended
- `qwen3.5-9b-qwen3.6-reasoning-distilled.q80.gguf`  (~9.5 GB) — higher fidelity
- `qwen3.5-9b-qwen3.6-reasoning-distilled.f16.gguf`  (~17.9 GB) — full precision

## How this adapter was trained

```bash
# Reproduce (end-to-end)
python app.py prepare
python app.py baseline
python app.py train
python app.py eval
python app.py export
```

All hyperparameters and the verification manifest live in the accompanying
`app.py` + `requirements.txt`. The held-out test set and GSM8K sample IDs are
locked in `data/splits/manifest.json` / `data/gsm8k_sample.jsonl`.

{tool_section}

## Limitations

- Only **500** training examples — sufficient for style/format transfer but not
  for large knowledge updates.
- Teacher (Qwen3.6-plus) quality is the ceiling. Any reasoning error in the
  teacher's traces can propagate to the student.
- GSM8K is one narrow probe; full reasoning capability should be re-evaluated
  on your downstream task.
- The LoRA targets only linear layers (not embeddings); the adapter preserves
  the base model's tokenizer and vocabulary.

## Files

- `adapter_config.json`, `adapter_model.safetensors` — LoRA adapter (PEFT).
- `*.gguf` — merged + quantized weights for llama.cpp / Ollama (Q4_K_M, Q8_0).
- `final_report.json` — machine-readable eval report.
- `data/splits/manifest.json` — split manifest with SHA256 of each split.
"""


def cmd_upload() -> None:
    """Upload adapter + merged model + GGUFs + model card to HF Hub.

    Requires HF_TOKEN in env. Does NOT accept token as CLI arg and does NOT log it.
    """
    _ensure_dirs()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN env var is not set. Do NOT pass as CLI arg. "
            "Run as:  HF_TOKEN=xxx .venv/bin/python app.py upload"
        )
    # From this point we never print or log the token value.

    repo_base = os.environ.get("DISTILL_HF_REPO", None)
    # Derive repo name from HF whoami if not overridden
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    me = api.whoami()["name"]
    repo_adapter = repo_base or f"{me}/{CONFIG['output_name']}"
    repo_gguf = f"{repo_adapter}-GGUF"

    # Load final report and generate card
    report_path = PATHS["reports"] / "final_report.json"
    if not report_path.exists():
        raise FileNotFoundError("Run `eval` first — final_report.json missing.")
    report = json.loads(report_path.read_text())
    card_text = _generate_model_card(report)

    # ---- Adapter repo (LoRA + merged + card + report)
    adapter_dir = PATHS["checkpoints"] / CONFIG["output_name"] / "final_adapter"
    merged_dir = PATHS["merged"] / CONFIG["output_name"]

    _log("upload", f"Creating repo {repo_adapter} (if missing)...")
    api.create_repo(repo_adapter, exist_ok=True, private=False, repo_type="model")

    # Write README.md into adapter dir and upload
    card_path = adapter_dir / "README.md"
    card_path.write_text(card_text)
    (adapter_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    # Also copy the split manifest for audit
    manifest_src = PATHS["splits"] / "manifest.json"
    if manifest_src.exists():
        (adapter_dir / "data_split_manifest.json").write_text(manifest_src.read_text())

    _log("upload", f"Uploading adapter dir → {repo_adapter} ...")
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_adapter,
        repo_type="model",
        commit_message="Upload LoRA adapter + eval report",
    )

    # Optionally upload merged full weights to the same repo under a subfolder
    if merged_dir.exists() and any(merged_dir.iterdir()):
        _log("upload", f"Uploading merged model → {repo_adapter}/merged/ ...")
        api.upload_folder(
            folder_path=str(merged_dir),
            repo_id=repo_adapter,
            repo_type="model",
            path_in_repo="merged",
            commit_message="Upload merged fp16 weights",
        )

    # ---- GGUF repo
    gguf_dir = PATHS["gguf"] / CONFIG["output_name"]
    if gguf_dir.exists() and any(gguf_dir.iterdir()):
        _log("upload", f"Creating repo {repo_gguf} (if missing)...")
        api.create_repo(repo_gguf, exist_ok=True, private=False, repo_type="model")
        (gguf_dir / "README.md").write_text(card_text)
        _log("upload", f"Uploading GGUFs → {repo_gguf} ...")
        api.upload_folder(
            folder_path=str(gguf_dir),
            repo_id=repo_gguf,
            repo_type="model",
            commit_message="Upload GGUF quantizations",
        )

    _log("upload", f"Done. Repos: https://huggingface.co/{repo_adapter}  and  https://huggingface.co/{repo_gguf}")


# ---------------------------------------------------------------------------
# Section 10: Top-level dispatch
# ---------------------------------------------------------------------------


def cmd_all() -> None:
    cmd_prepare()
    cmd_baseline()
    cmd_train()
    cmd_eval()
    cmd_export()
    # upload is opt-in and requires HF_TOKEN


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B reasoning distillation harness",
    )
    parser.add_argument(
        "cmd",
        choices=[
            "bootstrap",
            "prepare",
            "baseline",
            "train",
            "_train_worker",
            "eval",
            "export",
            "prepare-tools",
            "baseline-tools",
            "eval-tools",
            "verify-ollama",
            "upload",
            "all",
        ],
    )
    parser.add_argument("--force-venv", action="store_true")
    args = parser.parse_args()

    if args.cmd == "bootstrap":
        bootstrap(force=args.force_venv)
        print(f"venv ready: {VENV}")
        return

    # All other commands need the venv
    bootstrap(force=args.force_venv)

    if args.cmd == "prepare":
        cmd_prepare()
    elif args.cmd == "baseline":
        cmd_baseline()
    elif args.cmd == "train":
        cmd_train()
    elif args.cmd == "_train_worker":
        _train_worker()
    elif args.cmd == "eval":
        cmd_eval()
    elif args.cmd == "export":
        cmd_export()
    elif args.cmd == "prepare-tools":
        cmd_prepare_tools()
    elif args.cmd == "baseline-tools":
        cmd_baseline_tools()
    elif args.cmd == "eval-tools":
        cmd_eval_tools()
    elif args.cmd == "verify-ollama":
        cmd_verify_ollama()
    elif args.cmd == "upload":
        cmd_upload()
    elif args.cmd == "all":
        cmd_all()


if __name__ == "__main__":
    main()
