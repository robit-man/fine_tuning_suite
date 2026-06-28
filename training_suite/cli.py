from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_suite.core.config import DEFAULT_TARGET_CAPABILITIES, PATHS, PROJECT_ROOT, safe_model_tag, slugify
from training_suite.core.jobs import JobRunner
from training_suite.core.state import StateStore
from training_suite.evals.runner import capability_gate, tool_smoke
from training_suite.models.intake import inspect_intake
from training_suite.models.ollama import ModelfileSpec, generate_modelfile, show_model, write_modelfile


def cmd_web(args: argparse.Namespace) -> None:
    from training_suite.web import create_app

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


def cmd_db_init(_: argparse.Namespace) -> None:
    PATHS.ensure()
    store = StateStore()
    print(f"state ready: {store.db_path}")


def cmd_intake(args: argparse.Namespace) -> None:
    caps = args.target_capability or list(DEFAULT_TARGET_CAPABILITIES)
    result = inspect_intake(
        source=args.source,
        raw_source=args.raw_source,
        gguf_path=args.gguf_path,
        ollama_model=args.ollama_model,
        donor_model=args.donor_model,
        target_capabilities=caps,
    )
    data = result.to_model_row()
    if args.save:
        model_id = StateStore().upsert_model(data)
        data["id"] = model_id
    print(json.dumps(data, indent=2, sort_keys=True))


def cmd_ollama_show(args: argparse.Namespace) -> None:
    shown = show_model(args.model, verbose=args.verbose, include_modelfile=args.modelfile)
    print(json.dumps(shown.to_dict(), indent=2, sort_keys=True))


def cmd_modelfile(args: argparse.Namespace) -> None:
    params = {}
    for item in args.parameter or []:
        key, sep, value = item.partition("=")
        if sep:
            params[key] = value
    spec = ModelfileSpec(
        from_ref=args.from_ref,
        renderer=args.renderer,
        parser=args.parser,
        parameters=params,
        template=args.template,
        requires=args.requires,
    )
    text = generate_modelfile(spec)
    if args.out:
        write_modelfile(Path(args.out), spec)
    print(text)


def cmd_job_list(args: argparse.Namespace) -> None:
    jobs = StateStore().list_jobs(args.limit)
    print(json.dumps(jobs, indent=2, sort_keys=True))


def cmd_tool_smoke(args: argparse.Namespace) -> None:
    print(json.dumps(tool_smoke(args.model), indent=2, sort_keys=True))


def cmd_capability_gate(args: argparse.Namespace) -> None:
    print(json.dumps(capability_gate(args.model, args.capability), indent=2, sort_keys=True))


def cmd_ornith_seed(args: argparse.Namespace) -> None:
    """Register the canonical Ornith 9B test case without downloading weights."""
    result = inspect_intake(
        source="https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF",
        raw_source="https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B",
        donor_model=args.donor_model,
        ollama_model=args.ollama_model,
        target_capabilities=list(DEFAULT_TARGET_CAPABILITIES),
    )
    model_id = StateStore().upsert_model(result.to_model_row())
    print(json.dumps({"id": model_id, **result.to_model_row()}, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training Suite CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    web = sub.add_parser("web", help="Start the Flask dashboard")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=7860)
    web.add_argument("--debug", action="store_true")
    web.set_defaults(func=cmd_web)

    db = sub.add_parser("db-init", help="Initialize state database")
    db.set_defaults(func=cmd_db_init)

    intake = sub.add_parser("intake", help="Inspect and optionally save a model intake")
    intake.add_argument("--source", required=True)
    intake.add_argument("--raw-source")
    intake.add_argument("--gguf-path")
    intake.add_argument("--ollama-model")
    intake.add_argument("--donor-model")
    intake.add_argument("--target-capability", action="append")
    intake.add_argument("--save", action="store_true")
    intake.set_defaults(func=cmd_intake)

    show = sub.add_parser("ollama-show", help="Inspect a local Ollama model")
    show.add_argument("model")
    show.add_argument("--verbose", action="store_true")
    show.add_argument("--modelfile", action="store_true")
    show.set_defaults(func=cmd_ollama_show)

    mf = sub.add_parser("modelfile", help="Generate an Ollama Modelfile")
    mf.add_argument("--from", dest="from_ref", required=True)
    mf.add_argument("--renderer", default="qwen3.5")
    mf.add_argument("--parser", default="qwen3.5")
    mf.add_argument("--template", default="{{ .Prompt }}")
    mf.add_argument("--requires")
    mf.add_argument("--parameter", action="append", help="PARAMETER as key=value")
    mf.add_argument("--out")
    mf.set_defaults(func=cmd_modelfile)

    jobs = sub.add_parser("job-list", help="List tracked jobs")
    jobs.add_argument("--limit", type=int, default=20)
    jobs.set_defaults(func=cmd_job_list)

    smoke = sub.add_parser("tool-smoke", help="Run a synchronous Ollama tool smoke test")
    smoke.add_argument("model")
    smoke.set_defaults(func=cmd_tool_smoke)

    gate = sub.add_parser("capability-gate", help="Check Ollama advertised capabilities")
    gate.add_argument("model")
    gate.add_argument("--capability", action="append", default=["vision", "tools", "thinking"])
    gate.set_defaults(func=cmd_capability_gate)

    ornith = sub.add_parser("ornith-seed", help="Register the canonical Ornith 9B intake")
    ornith.add_argument("--donor-model", default="qwen3.5:9b")
    ornith.add_argument("--ollama-model")
    ornith.set_defaults(func=cmd_ornith_seed)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
