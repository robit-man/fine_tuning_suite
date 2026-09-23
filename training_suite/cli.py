from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_suite.core.config import DEFAULT_TARGET_CAPABILITIES, PATHS
from training_suite.core.state import StateStore
from training_suite.evals.runner import (
    capability_gate,
    omni_audio_smoke,
    tool_smoke,
    write_eval_report,
)
from training_suite.models.audio_bridge_projector import build_audio_bridge_projector
from training_suite.models.audio_bridge_release import (
    AudioBridgeReleaseSpec,
    build_audio_bridge_release,
)
from training_suite.models.intake import inspect_intake
from training_suite.models.ollama import (
    ModelfileSpec,
    generate_modelfile,
    show_model,
    write_modelfile,
)
from training_suite.models.ollama_audio_bridge import create_audio_bridge_tag
from training_suite.models.ollama_sidecar import (
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
    pack_audio_bridge_sidecar,
    pack_monolithic_gguf,
)
from training_suite.training.bridge_initialization import (
    initialize_bridge_from_token_spaces,
)
from training_suite.training.omni_encoder_bridge import (
    build_bridge_manifest,
    write_bridge_manifest,
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
    report = tool_smoke(args.model)
    if args.out:
        write_eval_report(Path(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("ok"):
        raise SystemExit(1)


def cmd_capability_gate(args: argparse.Namespace) -> None:
    report = capability_gate(args.model, args.capability)
    if args.out:
        write_eval_report(Path(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("ok"):
        raise SystemExit(1)


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


def cmd_omni_bridge_plan(args: argparse.Namespace) -> None:
    text_config = load_config_reference(args.text_source)
    omni_config = load_config_reference(args.omni_source)
    manifest = build_bridge_manifest(
        text_config=text_config,
        omni_config=omni_config,
        target_source=args.text_source,
        omni_source=args.omni_source,
    )
    if args.out:
        output = write_bridge_manifest(Path(args.out), manifest)
        manifest["output"] = str(output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


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
        views=tuple(args.view) if args.view else None,
        models_dir=Path(args.models_dir) if args.models_dir else None,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_omni_bridge_assemble(args: argparse.Namespace) -> None:
    report = build_audio_bridge_projector(
        base_projector_gguf=Path(args.base_projector),
        omni_projector_gguf=Path(args.omni_projector),
        bridge_checkpoint=Path(args.checkpoint),
        out_gguf=Path(args.out),
        target_name=args.name,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_omni_bridge_initialize(args: argparse.Namespace) -> None:
    report = initialize_bridge_from_token_spaces(
        omni_text_gguf=Path(args.omni_text_gguf),
        target_text_gguf=Path(args.target_text_gguf),
        omni_projector_gguf=Path(args.omni_projector),
        output_dir=Path(args.out),
        device=args.device,
        max_tokens=args.max_tokens,
        holdout_tokens=args.holdout_tokens,
        ridge=args.ridge,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_omni_bridge_pack(args: argparse.Namespace) -> None:
    report = pack_audio_bridge_sidecar(
        tts_gguf=Path(args.tts_gguf),
        tts_projector_gguf=Path(args.tts_projector_gguf),
        out_gguf=Path(args.out),
        base_source=args.base_source,
        combined_projector_source=args.combined_projector_source,
        tts_source=args.tts_source,
        tts_projector_source=args.tts_projector_source,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_omni_bridge_tag(args: argparse.Namespace) -> None:
    report = create_audio_bridge_tag(
        source_model=args.source_model,
        target_model=args.target_model,
        combined_projector_gguf=Path(args.combined_projector),
        tts_sidecar_gguf=Path(args.tts_sidecar),
        models_dir=Path(args.models_dir) if args.models_dir else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_omni_bridge_release(args: argparse.Namespace) -> None:
    report = build_audio_bridge_release(
        spec=AudioBridgeReleaseSpec(
            display_name=args.name,
            base_model=args.base_model,
            ollama_tag=args.ollama_tag,
            classifier=args.classifier,
            quantization=args.quantization,
            prior_bundle_bytes=args.prior_bundle_bytes,
            license_id=args.license,
            license_name=args.license_name,
            component_models=tuple(args.component_model),
        ),
        language_model_gguf=Path(args.language_model),
        language_filename=args.language_filename,
        combined_projector_gguf=Path(args.combined_projector),
        projector_filename=args.projector_filename,
        tts_sidecar_gguf=Path(args.tts_sidecar),
        sidecar_filename=args.sidecar_filename,
        training_report=Path(args.training_report),
        evaluation_report=Path(args.evaluation_report),
        vision_evaluation_report=Path(args.vision_evaluation_report),
        capability_report=Path(args.capability_report),
        tool_smoke_report=Path(args.tool_smoke_report),
        tts_alignment_report=Path(args.tts_alignment_report),
        tts_vocabulary_report=Path(args.tts_vocabulary_report),
        tts_vocabulary_audit_report=Path(args.tts_vocabulary_audit_report),
        output_dir=Path(args.out),
        checkpoint_selection_report=(
            Path(args.checkpoint_selection_report)
            if args.checkpoint_selection_report
            else None
        ),
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
    smoke.add_argument("--out")
    smoke.set_defaults(func=cmd_tool_smoke)

    gate = sub.add_parser("capability-gate", help="Check Ollama advertised capabilities")
    gate.add_argument("model")
    gate.add_argument("--capability", action="append", default=["vision", "tools", "thinking"])
    gate.add_argument("--out")
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

    omni_bridge = sub.add_parser(
        "omni-bridge-plan",
        help="Plan a frozen Omni-audio to frozen language-model embedding bridge",
    )
    omni_bridge.add_argument(
        "--text-source",
        required=True,
        help="Local config.json path or Hugging Face repo for the frozen language model",
    )
    omni_bridge.add_argument(
        "--omni-source",
        default=QWEN3_OMNI_INSTRUCT,
        help="Local config.json path or Hugging Face repo for the frozen Omni encoder",
    )
    omni_bridge.add_argument("--out", help="Write the bridge experiment manifest here")
    omni_bridge.set_defaults(func=cmd_omni_bridge_plan)

    omni_bridge_assemble = sub.add_parser(
        "omni-bridge-assemble",
        help="Build a target-native vision plus trained Omni-audio projector GGUF",
    )
    omni_bridge_assemble.add_argument("--base-projector", required=True)
    omni_bridge_assemble.add_argument("--omni-projector", required=True)
    omni_bridge_assemble.add_argument("--checkpoint", required=True)
    omni_bridge_assemble.add_argument("--out", required=True)
    omni_bridge_assemble.add_argument("--name", required=True)
    omni_bridge_assemble.add_argument("--overwrite", action="store_true")
    omni_bridge_assemble.set_defaults(func=cmd_omni_bridge_assemble)

    omni_bridge_initialize = sub.add_parser(
        "omni-bridge-initialize",
        help="Initialize a deployable audio bridge from shared tokenizer embeddings",
    )
    omni_bridge_initialize.add_argument("--omni-text-gguf", required=True)
    omni_bridge_initialize.add_argument("--target-text-gguf", required=True)
    omni_bridge_initialize.add_argument("--omni-projector", required=True)
    omni_bridge_initialize.add_argument("--out", required=True)
    omni_bridge_initialize.add_argument("--device", default="cpu")
    omni_bridge_initialize.add_argument("--max-tokens", type=int, default=32768)
    omni_bridge_initialize.add_argument("--holdout-tokens", type=int, default=2048)
    omni_bridge_initialize.add_argument("--ridge", type=float, default=1e-3)
    omni_bridge_initialize.add_argument("--seed", type=int, default=42)
    omni_bridge_initialize.set_defaults(func=cmd_omni_bridge_initialize)

    omni_bridge_pack = sub.add_parser(
        "omni-bridge-pack",
        help="Pack the lightweight TTS sidecar for a trained audio-bridge model",
    )
    omni_bridge_pack.add_argument("--tts-gguf", required=True)
    omni_bridge_pack.add_argument("--tts-projector-gguf", required=True)
    omni_bridge_pack.add_argument("--base-source", required=True)
    omni_bridge_pack.add_argument("--combined-projector-source", required=True)
    omni_bridge_pack.add_argument("--tts-source")
    omni_bridge_pack.add_argument("--tts-projector-source")
    omni_bridge_pack.add_argument("--out", required=True)
    omni_bridge_pack.add_argument("--overwrite", action="store_true")
    omni_bridge_pack.set_defaults(func=cmd_omni_bridge_pack)

    omni_bridge_tag = sub.add_parser(
        "omni-bridge-tag",
        help="Create an Ollama tag with a trained combined projector and TTS sidecar",
    )
    omni_bridge_tag.add_argument("--source-model", required=True)
    omni_bridge_tag.add_argument("--target-model", required=True)
    omni_bridge_tag.add_argument("--combined-projector", required=True)
    omni_bridge_tag.add_argument("--tts-sidecar", required=True)
    omni_bridge_tag.add_argument("--models-dir")
    omni_bridge_tag.set_defaults(func=cmd_omni_bridge_tag)

    omni_bridge_release = sub.add_parser(
        "omni-bridge-release",
        help="Build gated release evidence and a Hugging Face model card",
    )
    omni_bridge_release.add_argument("--name", required=True)
    omni_bridge_release.add_argument("--base-model", required=True)
    omni_bridge_release.add_argument("--ollama-tag", required=True)
    omni_bridge_release.add_argument("--classifier", default="audio-bridge")
    omni_bridge_release.add_argument("--license", default="other")
    omni_bridge_release.add_argument("--license-name")
    omni_bridge_release.add_argument(
        "--component-model",
        action="append",
        default=[],
        help="Repeat for every upstream model represented by the release",
    )
    omni_bridge_release.add_argument(
        "--quantization",
        default="Q4_K_M language; BF16 final audio projection",
    )
    omni_bridge_release.add_argument("--prior-bundle-bytes", required=True, type=int)
    omni_bridge_release.add_argument("--language-model", required=True)
    omni_bridge_release.add_argument("--language-filename", required=True)
    omni_bridge_release.add_argument("--combined-projector", required=True)
    omni_bridge_release.add_argument("--projector-filename", required=True)
    omni_bridge_release.add_argument("--tts-sidecar", required=True)
    omni_bridge_release.add_argument("--sidecar-filename", required=True)
    omni_bridge_release.add_argument("--training-report", required=True)
    omni_bridge_release.add_argument("--evaluation-report", required=True)
    omni_bridge_release.add_argument("--vision-evaluation-report", required=True)
    omni_bridge_release.add_argument("--capability-report", required=True)
    omni_bridge_release.add_argument("--tool-smoke-report", required=True)
    omni_bridge_release.add_argument("--tts-alignment-report", required=True)
    omni_bridge_release.add_argument("--tts-vocabulary-report", required=True)
    omni_bridge_release.add_argument(
        "--tts-vocabulary-audit-report",
        required=True,
        help="Strict audit binding the complete vocabulary report",
    )
    omni_bridge_release.add_argument(
        "--checkpoint-selection-report",
        help="Optional behavior-gated checkpoint selection evidence",
    )
    omni_bridge_release.add_argument("--out", required=True)
    omni_bridge_release.set_defaults(func=cmd_omni_bridge_release)

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
