from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_suite.core.config import DEFAULT_TARGET_CAPABILITIES, PATHS
from training_suite.core.state import StateStore
from training_suite.evals.runner import capability_gate, omni_audio_smoke, tool_smoke
from training_suite.models.intake import inspect_intake
from training_suite.models.ollama import (
    ModelfileSpec,
    generate_modelfile,
    show_model,
    write_modelfile,
)
from training_suite.models.ollama_sidecar import (
    RUNTIME_VIEWS,
    attach_ollama_sidecar,
    prepare_ollama_sidecar,
    resolve_ollama_sidecar,
)
from training_suite.models.omni import (
    QWEN3_OMNI_INSTRUCT,
    load_config_reference,
    plan_omni_bundle,
    write_omni_bundle,
)
from training_suite.models.single_gguf import (
    inspect_monolithic_gguf,
    materialize_component_view,
    pack_monolithic_gguf,
)


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


def cmd_omni_audio_smoke(args: argparse.Namespace) -> None:
    report = omni_audio_smoke(
        Path(args.audio),
        endpoint=args.endpoint,
        prompt=args.prompt,
        timeout=args.timeout,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("ok"):
        raise SystemExit(1)


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


def cmd_omni_plan(args: argparse.Namespace) -> None:
    text_config = load_config_reference(args.text_source)
    omni_config = load_config_reference(args.omni_source)
    plan = plan_omni_bundle(
        text_config=text_config,
        omni_config=omni_config,
        text_source=args.text_source,
        omni_source=args.omni_source,
        target_tag=args.target_tag,
    )
    if args.out:
        plan["outputs"] = write_omni_bundle(
            Path(args.out),
            plan,
            text_gguf=args.text_gguf,
            mmproj_gguf=args.mmproj_gguf,
            talker_gguf=args.talker_gguf,
            code2wav_gguf=args.code2wav_gguf,
            renderer=args.renderer,
            parser=args.parser,
        )
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.require_native and plan["mode"] != "native-omni":
        raise SystemExit(2)


def cmd_omni_pack(args: argparse.Namespace) -> None:
    out = Path(args.out).expanduser().resolve()
    report = pack_monolithic_gguf(
        base_gguf=Path(args.base_gguf),
        base_projector_gguf=Path(args.base_projector_gguf) if args.base_projector_gguf else None,
        comprehension_gguf=Path(args.comprehension_gguf),
        comprehension_projector_gguf=(
            Path(args.comprehension_projector_gguf)
            if args.comprehension_projector_gguf
            else None
        ),
        tts_gguf=Path(args.tts_gguf),
        tts_projector_gguf=Path(args.tts_projector_gguf) if args.tts_projector_gguf else None,
        out_gguf=out,
        base_source=args.base_source,
        base_projector_source=args.base_projector_source,
        comprehension_source=args.comprehension_source,
        comprehension_projector_source=args.comprehension_projector_source,
        tts_source=args.tts_source,
        tts_projector_source=args.tts_projector_source,
        overwrite=args.overwrite,
    )
    report_path = out.with_suffix(out.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    layer_descriptor = out.with_name("ollama-sidecar-layer.json")
    layer_descriptor.write_text(
        json.dumps(
            {
                "mediaType": "application/vnd.robit.ollama.omni.bundle.v1+gguf",
                "path": str(out),
                "size": out.stat().st_size,
                "note": (
                    "Attach this sidecar to a stock-runnable Ollama manifest; "
                    "do not use it as a Modelfile FROM target."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report["report"] = str(report_path)
    report["ollama_sidecar_layer"] = str(layer_descriptor)
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_omni_inspect(args: argparse.Namespace) -> None:
    report = inspect_monolithic_gguf(Path(args.gguf))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


def cmd_omni_unpack(args: argparse.Namespace) -> None:
    report = materialize_component_view(
        bundle_gguf=Path(args.gguf),
        view=args.view,
        out_gguf=Path(args.out),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_omni_attach(args: argparse.Namespace) -> None:
    report = attach_ollama_sidecar(
        model=args.model,
        bundle_gguf=Path(args.gguf),
        models_dir=Path(args.models_dir) if args.models_dir else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_omni_resolve(args: argparse.Namespace) -> None:
    report = resolve_ollama_sidecar(
        model=args.model,
        models_dir=Path(args.models_dir) if args.models_dir else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_omni_prepare(args: argparse.Namespace) -> None:
    report = prepare_ollama_sidecar(
        model=args.model,
        output_dir=Path(args.out),
        views=tuple(args.view) if args.view else RUNTIME_VIEWS,
        models_dir=Path(args.models_dir) if args.models_dir else None,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


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

    omni = sub.add_parser(
        "omni-plan",
        help="Plan a native graft or one-file custom Ollama multimodal router",
    )
    omni.add_argument(
        "--text-source",
        required=True,
        help="Local config.json path or Hugging Face repo for the language model",
    )
    omni.add_argument(
        "--omni-source",
        default=QWEN3_OMNI_INSTRUCT,
        help="Local config.json path or Hugging Face repo for the Omni donor",
    )
    omni.add_argument("--target-tag")
    omni.add_argument("--out", help="Write omni_bundle.json and audio_contract.json here")
    omni.add_argument("--text-gguf", help="Ollama-compatible language GGUF")
    omni.add_argument("--mmproj-gguf", help="Audio/vision projector GGUF")
    omni.add_argument("--talker-gguf", help="Speech Talker GGUF")
    omni.add_argument("--code2wav-gguf", help="Codec-to-waveform GGUF")
    omni.add_argument("--renderer", default="qwen3.8")
    omni.add_argument("--parser", default="qwen3.5")
    omni.add_argument(
        "--require-native",
        action="store_true",
        help="Exit 2 when the text trunk cannot be substituted into the Omni Thinker",
    )
    omni.set_defaults(func=cmd_omni_plan)

    omni_pack = sub.add_parser(
        "omni-pack",
        help="Pack language, comprehension, and TTS GGUFs into one custom Ollama sidecar",
    )
    omni_pack.add_argument("--base-gguf", required=True, help="Qwen3.8/Ornith Ollama-compatible GGUF")
    omni_pack.add_argument("--base-projector-gguf", help="Original base vision projector GGUF")
    omni_pack.add_argument(
        "--comprehension-gguf",
        required=True,
        help="Self-contained audio/video understanding GGUF",
    )
    omni_pack.add_argument(
        "--comprehension-projector-gguf",
        help="Audio/vision projector for the comprehension model",
    )
    omni_pack.add_argument("--tts-gguf", required=True, help="Text-conditioned TTS GGUF")
    omni_pack.add_argument("--tts-projector-gguf", help="TTS codec/waveform projector GGUF")
    omni_pack.add_argument("--out", required=True, help="Output namespaced sidecar .gguf path")
    omni_pack.add_argument("--base-source", help="Provenance label for the base model")
    omni_pack.add_argument("--base-projector-source", help="Provenance label for base projector")
    omni_pack.add_argument("--comprehension-source", help="Provenance label for the comprehension model")
    omni_pack.add_argument(
        "--comprehension-projector-source",
        help="Provenance label for the comprehension projector",
    )
    omni_pack.add_argument("--tts-source", help="Provenance label for the TTS model")
    omni_pack.add_argument("--tts-projector-source", help="Provenance label for TTS projector")
    omni_pack.add_argument("--renderer", default="qwen3.8")
    omni_pack.add_argument("--parser", default="qwen3.5")
    omni_pack.add_argument("--requires", help="Minimum custom Ollama build version")
    omni_pack.add_argument("--num-ctx", type=int, default=262144)
    omni_pack.add_argument("--overwrite", action="store_true")
    omni_pack.set_defaults(func=cmd_omni_pack)

    omni_inspect = sub.add_parser(
        "omni-inspect",
        help="Inspect and validate a namespaced audio/video/TTS GGUF sidecar",
    )
    omni_inspect.add_argument("gguf")
    omni_inspect.set_defaults(func=cmd_omni_inspect)

    omni_unpack = sub.add_parser(
        "omni-unpack",
        help="Materialize one executable model/projector view from a sidecar",
    )
    omni_unpack.add_argument("gguf")
    omni_unpack.add_argument(
        "--view",
        required=True,
        choices=[
            "base",
            "base_projector",
            "comprehension_model",
            "comprehension_projector",
            "tts_model",
            "tts_projector",
        ],
    )
    omni_unpack.add_argument("--out", required=True)
    omni_unpack.add_argument("--overwrite", action="store_true")
    omni_unpack.set_defaults(func=cmd_omni_unpack)

    omni_attach = sub.add_parser(
        "omni-attach",
        help="Attach an Omni GGUF sidecar layer to an existing runnable Ollama tag",
    )
    omni_attach.add_argument("model", help="Existing local Ollama tag")
    omni_attach.add_argument("gguf", help="Validated namespaced Omni GGUF")
    omni_attach.add_argument("--models-dir", help="Override the Ollama model store")
    omni_attach.set_defaults(func=cmd_omni_attach)

    omni_resolve = sub.add_parser(
        "omni-resolve",
        help="Resolve and inspect the Omni sidecar attached to a local Ollama tag",
    )
    omni_resolve.add_argument("model")
    omni_resolve.add_argument("--models-dir", help="Override the Ollama model store")
    omni_resolve.set_defaults(func=cmd_omni_resolve)

    omni_prepare = sub.add_parser(
        "omni-prepare",
        help="Materialize disposable media-runtime views from an installed Omni tag",
    )
    omni_prepare.add_argument("model")
    omni_prepare.add_argument("--out", required=True, help="Disposable component cache directory")
    omni_prepare.add_argument(
        "--view",
        action="append",
        choices=[
            "comprehension_model",
            "comprehension_projector",
            "tts_model",
            "tts_projector",
        ],
        help="Runtime view to materialize; repeat to select a subset",
    )
    omni_prepare.add_argument("--models-dir", help="Override the Ollama model store")
    omni_prepare.add_argument("--overwrite", action="store_true")
    omni_prepare.set_defaults(func=cmd_omni_prepare)

    audio_smoke = sub.add_parser(
        "omni-audio-smoke",
        help="Run a live audio-in/text/audio-out cascade probe",
    )
    audio_smoke.add_argument("--audio", required=True, help="16 kHz mono PCM16 WAV fixture")
    audio_smoke.add_argument(
        "--endpoint",
        default="http://127.0.0.1:7860/api/omni/cascade",
    )
    audio_smoke.add_argument(
        "--prompt",
        default="Transcribe this audio and answer naturally.",
    )
    audio_smoke.add_argument("--timeout", type=float, default=900)
    audio_smoke.set_defaults(func=cmd_omni_audio_smoke)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
