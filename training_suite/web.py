from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from training_suite.core.config import (
    DEFAULT_PARSER,
    DEFAULT_RENDERER,
    DEFAULT_TARGET_CAPABILITIES,
    PATHS,
    PROJECT_ROOT,
    TARGET_CAPABILITIES,
    safe_model_tag,
    slugify,
)
from training_suite.core.jobs import JobRunner
from training_suite.core.state import StateStore
from training_suite.datasets.registry import CURATION_RECIPES, dataset_record
from training_suite.evals.runner import (
    capability_gate,
    eval_specs,
    get_eval,
    tool_smoke,
    write_eval_report,
)
from training_suite.models.intake import inspect_intake
from training_suite.models.audio import (
    AudioContractError,
    DEFAULT_AUDIO_CONTRACT,
    validate_audio_input,
)
from training_suite.models.omni import (
    QWEN3_OMNI_INSTRUCT,
    load_config_reference,
    plan_omni_bundle,
)
from training_suite.models.single_gguf import audio_router_contract
from training_suite.omni_runtime import (
    OmniRuntimeConfig,
    OmniRuntimeError,
    run_http_cascade,
)
from training_suite.models.ollama import (
    ModelfileSpec,
    copy_command,
    create_command,
    generate_modelfile,
    push_command,
    show_model,
    signin_command,
    write_modelfile,
)
from training_suite.training.adapters import action_specs, get_action


def _router_audio_and_prompt(data: dict[str, Any]) -> tuple[Any, str]:
    audio_payload = data.get("audio")
    prompt = str(data.get("prompt") or "")
    if audio_payload is None and isinstance(data.get("messages"), list):
        for message in reversed(data["messages"]):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            audios = message.get("audios")
            if isinstance(audios, list) and audios:
                audio_payload = audios[0]
            if not prompt and isinstance(message.get("content"), str):
                prompt = message["content"]
            if audio_payload is not None:
                break
    return audio_payload, prompt


def create_app(
    store: StateStore | None = None, runner: JobRunner | None = None
) -> Flask:
    PATHS.ensure()
    app = Flask(__name__)
    app.secret_key = "training-suite-local-dashboard"
    app.config["STORE"] = store or StateStore()
    app.config["RUNNER"] = runner or JobRunner(app.config["STORE"])

    @app.template_filter("pretty_json")
    def pretty_json(value: Any) -> str:
        return json.dumps(value, indent=2, sort_keys=True)

    @app.template_filter("cap_class")
    def cap_class(value: str) -> str:
        return "cap cap-" + slugify(value)

    @app.context_processor
    def inject_globals() -> dict[str, Any]:
        return {
            "target_capabilities": TARGET_CAPABILITIES,
            "default_target_capabilities": DEFAULT_TARGET_CAPABILITIES,
        }

    @app.get("/")
    def dashboard() -> str:
        store: StateStore = app.config["STORE"]
        models = store.list_models()
        jobs = store.list_jobs(10)
        eval_runs = store.list_eval_runs(10)
        return render_template(
            "dashboard.html", models=models, jobs=jobs, eval_runs=eval_runs
        )

    @app.route("/intake", methods=["GET", "POST"])
    def intake() -> str:
        store: StateStore = app.config["STORE"]
        result = None
        if request.method == "POST":
            caps = request.form.getlist("target_capabilities") or list(
                DEFAULT_TARGET_CAPABILITIES
            )
            result = inspect_intake(
                source=request.form.get("source", ""),
                raw_source=request.form.get("raw_source") or None,
                gguf_path=request.form.get("gguf_path") or None,
                ollama_model=request.form.get("ollama_model") or None,
                donor_model=request.form.get("donor_model") or None,
                target_capabilities=caps,
            )
            model_id = store.upsert_model(result.to_model_row())
            flash(f"Intake saved as model #{model_id}.", "success")
            return redirect(url_for("model_detail", model_id=model_id))
        return render_template("intake.html", result=result)

    @app.get("/models/<int:model_id>")
    def model_detail(model_id: int) -> str:
        store: StateStore = app.config["STORE"]
        model = store.get_model(model_id)
        if not model:
            flash("Model not found.", "error")
            return redirect(url_for("dashboard"))
        return render_template(
            "model_detail.html", model=model, jobs=store.list_jobs(50)
        )

    @app.route("/datasets", methods=["GET", "POST"])
    def datasets() -> str:
        store: StateStore = app.config["STORE"]
        if request.method == "POST":
            mapping = _json_form("schema_mapping")
            split_config = _json_form("split_config")
            record = dataset_record(
                name=request.form.get("name", ""),
                source=request.form.get("source", ""),
                schema_mapping=mapping,
                split_config=split_config,
                license_note=request.form.get("license_note") or None,
            )
            dataset_id = store.add_dataset(record)
            flash(f"Dataset registered as #{dataset_id}.", "success")
            return redirect(url_for("datasets"))
        return render_template(
            "datasets.html",
            datasets=store.list_datasets(),
            recipes=CURATION_RECIPES,
        )

    @app.route("/actions", methods=["GET", "POST"])
    def actions() -> str:
        store: StateStore = app.config["STORE"]
        runner: JobRunner = app.config["RUNNER"]
        if request.method == "POST":
            key = request.form.get("action", "")
            model_id = _int_or_none(request.form.get("model_id"))
            dataset_id = _int_or_none(request.form.get("dataset_id"))
            try:
                action = get_action(key)
            except KeyError as exc:
                flash(str(exc), "error")
                return redirect(url_for("actions"))
            job_id = runner.start(
                kind=action.kind,
                command=action.command,
                cwd=PATHS.package_root,
                model_id=model_id,
                dataset_id=dataset_id,
                metadata={"action": action.key, "label": action.label},
            )
            flash(f"Started job #{job_id}: {action.label}.", "success")
            return redirect(url_for("job_detail", job_id=job_id))
        return render_template(
            "actions.html",
            actions=action_specs().values(),
            models=store.list_models(),
            datasets=store.list_datasets(),
            jobs=store.list_jobs(30),
        )

    @app.get("/jobs/<int:job_id>")
    def job_detail(job_id: int) -> str:
        store: StateStore = app.config["STORE"]
        runner: JobRunner = app.config["RUNNER"]
        job = store.get_job(job_id)
        if not job:
            flash("Job not found.", "error")
            return redirect(url_for("actions"))
        return render_template("job.html", job=job, log=runner.read_log(job_id))

    @app.get("/api/jobs/<int:job_id>")
    def job_api(job_id: int) -> Response:
        store: StateStore = app.config["STORE"]
        runner: JobRunner = app.config["RUNNER"]
        job = store.get_job(job_id)
        if not job:
            return jsonify({"error": "not found"}), 404
        return jsonify({"job": job, "log": runner.read_log(job_id)})

    @app.post("/jobs/<int:job_id>/cancel")
    def cancel_job(job_id: int) -> Response:
        runner: JobRunner = app.config["RUNNER"]
        ok = runner.cancel(job_id)
        if request.is_json or request.accept_mimetypes.best == "application/json":
            return jsonify({"cancelled": ok})
        flash(
            "Cancel requested." if ok else "Job is not running in this process.", "info"
        )
        return redirect(url_for("job_detail", job_id=job_id))

    @app.route("/export", methods=["GET", "POST"])
    def export() -> str:
        store: StateStore = app.config["STORE"]
        runner: JobRunner = app.config["RUNNER"]
        generated = None
        if request.method == "POST":
            action = request.form.get("export_action", "generate")
            model_name = safe_model_tag(request.form.get("model_name", "model:latest"))
            from_ref = request.form.get("from_ref", "").strip()
            model_id = _int_or_none(request.form.get("model_id"))
            spec = _modelfile_spec_from_form()
            out_dir = (
                PATHS.outputs
                / "ollama"
                / slugify(model_name.replace("/", "-").replace(":", "-"))
            )
            modelfile_path = out_dir / "Modelfile"
            if action == "generate":
                generated = generate_modelfile(spec)
                write_modelfile(modelfile_path, spec)
                flash(f"Modelfile written to {modelfile_path}.", "success")
            elif action == "create":
                write_modelfile(modelfile_path, spec)
                job_id = runner.start(
                    kind="ollama-create",
                    command=create_command(model_name, modelfile_path),
                    cwd=out_dir,
                    model_id=model_id,
                    metadata={"model_name": model_name, "from_ref": from_ref},
                )
                flash(f"Started Ollama create job #{job_id}.", "success")
                return redirect(url_for("job_detail", job_id=job_id))
            elif action == "signin":
                job_id = runner.start(
                    kind="ollama-signin", command=signin_command(), cwd=PROJECT_ROOT
                )
                flash(f"Started terminal-assisted sign-in job #{job_id}.", "info")
                return redirect(url_for("job_detail", job_id=job_id))
            elif action == "copy-push":
                remote = safe_model_tag(request.form.get("remote_model", ""))
                if not remote or remote.startswith("model:"):
                    flash("Remote model tag is required for upload.", "error")
                else:
                    cp_id = runner.start(
                        kind="ollama-copy",
                        command=copy_command(model_name, remote),
                        cwd=PROJECT_ROOT,
                        model_id=model_id,
                    )
                    push_id = runner.start(
                        kind="ollama-push",
                        command=push_command(remote),
                        cwd=PROJECT_ROOT,
                        model_id=model_id,
                    )
                    flash(
                        f"Started copy job #{cp_id} and push job #{push_id}.", "success"
                    )
                    return redirect(url_for("actions"))
        return render_template(
            "export.html",
            models=store.list_models(),
            generated=generated,
            default_renderer=DEFAULT_RENDERER,
            default_parser=DEFAULT_PARSER,
        )

    @app.route("/evaluation", methods=["GET", "POST"])
    def evaluation() -> str:
        store: StateStore = app.config["STORE"]
        runner: JobRunner = app.config["RUNNER"]
        sync_report = None
        model_name = request.values.get("model_name", "")
        if request.method == "POST":
            model_name = request.form.get("model_name", "").strip()
            model_id = _int_or_none(request.form.get("model_id"))
            eval_key = request.form.get("eval_key", "")
            if eval_key == "capability-gate":
                sync_report = capability_gate(model_name)
                report_path = PATHS.logs / f"capability_gate_{slugify(model_name)}.json"
                write_eval_report(report_path, sync_report)
                store.add_eval_run(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "eval_type": eval_key,
                        "status": "succeeded" if sync_report.get("ok") else "failed",
                        "report_path": str(report_path),
                        "metrics": sync_report,
                    }
                )
            elif eval_key == "tool-smoke-sync":
                sync_report = tool_smoke(model_name)
                report_path = PATHS.logs / f"tool_smoke_{slugify(model_name)}.json"
                write_eval_report(report_path, sync_report)
                store.add_eval_run(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "eval_type": eval_key,
                        "status": "succeeded" if sync_report.get("ok") else "failed",
                        "report_path": str(report_path),
                        "metrics": sync_report,
                    }
                )
            else:
                try:
                    spec = get_eval(eval_key, model_name)
                except KeyError as exc:
                    flash(str(exc), "error")
                    return redirect(url_for("evaluation"))
                job_id = runner.start(
                    kind=f"eval-{spec.key}",
                    command=spec.command,
                    cwd=PATHS.package_root,
                    model_id=model_id,
                    metadata={"eval": spec.key, "model_name": model_name},
                )
                store.add_eval_run(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "eval_type": spec.key,
                        "status": "running",
                    }
                )
                flash(f"Started eval job #{job_id}: {spec.label}.", "success")
                return redirect(url_for("job_detail", job_id=job_id))
        specs = eval_specs(model_name or "model:latest")
        return render_template(
            "evaluation.html",
            models=store.list_models(),
            eval_specs=specs.values(),
            eval_runs=store.list_eval_runs(50),
            sync_report=sync_report,
        )

    # -----------------------------------------------------------------------
    # RESTful API endpoints (for agent/MCP toolkit use)
    # -----------------------------------------------------------------------

    @app.get("/api/models")
    def api_models() -> Response:
        store: StateStore = app.config["STORE"]
        return jsonify(store.list_models())

    @app.get("/api/models/<int:model_id>")
    def api_model_get(model_id: int) -> Response:
        store: StateStore = app.config["STORE"]
        model = store.get_model(model_id)
        if not model:
            return jsonify({"error": "not found"}), 404
        return jsonify(model)

    @app.get("/api/omni/audio/contract")
    def api_omni_audio_contract() -> Response:
        return jsonify(DEFAULT_AUDIO_CONTRACT.to_dict())

    @app.get("/api/omni/router/contract")
    def api_omni_router_contract() -> Response:
        return jsonify(audio_router_contract())

    @app.post("/api/omni/audio/validate")
    def api_omni_audio_validate() -> Response:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            payload, _ = _router_audio_and_prompt(data)
            if payload is None:
                payload = data
        else:
            payload = data
        try:
            audio = validate_audio_input(payload)
        except AudioContractError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "audio": audio.metadata()})

    @app.post("/api/omni/plan")
    def api_omni_plan() -> Response:
        data = request.get_json(force=True)
        text_source = str(data.get("text_source") or "").strip()
        omni_source = str(data.get("omni_source") or QWEN3_OMNI_INSTRUCT).strip()
        try:
            text_config = data.get("text_config") or load_config_reference(text_source)
            omni_config = data.get("omni_config") or load_config_reference(omni_source)
            if not isinstance(text_config, dict) or not isinstance(omni_config, dict):
                raise ValueError("text_config and omni_config must be JSON objects")
            plan = plan_omni_bundle(
                text_config=text_config,
                omni_config=omni_config,
                text_source=text_source or "inline:text_config",
                omni_source=omni_source,
                target_tag=data.get("target_tag"),
            )
        except (ValueError, httpx.HTTPError) as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify(plan)

    @app.post("/api/omni/cascade")
    def api_omni_cascade() -> Response:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        audio_payload, prompt = _router_audio_and_prompt(data)
        speech_mode = str(data.get("speech_mode") or "auto").lower()
        if speech_mode not in {"auto", "always", "never"}:
            return jsonify({"error": "speech_mode must be auto, always, or never"}), 400
        response_modalities = data.get("response_modalities")
        wants_audio = (
            True
            if response_modalities is None
            else isinstance(response_modalities, list) and "audio" in response_modalities
        )
        synthesize = speech_mode == "always" or (speech_mode == "auto" and wants_audio)
        try:
            config = OmniRuntimeConfig.from_environment(require_tts=synthesize)
            report = run_http_cascade(
                audio_payload=audio_payload,
                prompt=prompt,
                config=config,
                synthesize=synthesize,
            )
        except AudioContractError as exc:
            return jsonify({"error": str(exc)}), 400
        except OmniRuntimeError as exc:
            status = 503 if "not configured" in str(exc) else 502
            return jsonify({"error": str(exc)}), status
        return jsonify(report)

    @app.post("/api/models")
    def api_model_create() -> Response:
        store: StateStore = app.config["STORE"]
        data = request.get_json(force=True)
        caps = data.get("target_capabilities") or list(DEFAULT_TARGET_CAPABILITIES)
        result = inspect_intake(
            source=data.get("source", ""),
            raw_source=data.get("raw_source"),
            gguf_path=data.get("gguf_path"),
            ollama_model=data.get("ollama_model"),
            donor_model=data.get("donor_model"),
            target_capabilities=caps,
        )
        model_id = store.upsert_model(result.to_model_row())
        return jsonify({"id": model_id, "model": result.to_model_row()}), 201

    @app.get("/api/jobs")
    def api_jobs() -> Response:
        store: StateStore = app.config["STORE"]
        limit = request.args.get("limit", 50, type=int)
        return jsonify(store.list_jobs(limit))

    # GET /api/jobs/<id> is handled by job_api() above (returns 404 on not found)

    @app.post("/api/jobs/<int:job_id>/cancel")
    def api_job_cancel(job_id: int) -> Response:
        runner: JobRunner = app.config["RUNNER"]
        ok = runner.cancel(job_id)
        return jsonify({"cancelled": ok})

    @app.post("/api/jobs")
    def api_job_create() -> Response:
        store: StateStore = app.config["STORE"]
        runner: JobRunner = app.config["RUNNER"]
        data = request.get_json(force=True)
        key = data.get("action", "")
        model_id = data.get("model_id")
        dataset_id = data.get("dataset_id")
        try:
            action = get_action(key)
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 400
        job_id = runner.start(
            kind=action.kind,
            command=action.command,
            cwd=PATHS.package_root,
            model_id=model_id,
            dataset_id=dataset_id,
            metadata={"action": action.key, "label": action.label},
        )
        return jsonify({"id": job_id, "kind": action.kind, "label": action.label}), 201

    # POST /api/jobs/<id>/cancel is handled by cancel_job() above (returns JSON)

    @app.get("/api/datasets")
    def api_datasets() -> Response:
        store: StateStore = app.config["STORE"]
        return jsonify(store.list_datasets())

    @app.post("/api/datasets")
    def api_dataset_create() -> Response:
        store: StateStore = app.config["STORE"]
        data = request.get_json(force=True)
        record = dataset_record(
            name=data.get("name", ""),
            source=data.get("source", ""),
            schema_mapping=data.get("schema_mapping", {}),
            split_config=data.get("split_config", {}),
            license_note=data.get("license_note"),
        )
        dataset_id = store.add_dataset(record)
        return jsonify({"id": dataset_id}), 201

    @app.get("/api/evals")
    def api_evals() -> Response:
        store: StateStore = app.config["STORE"]
        limit = request.args.get("limit", 50, type=int)
        return jsonify(store.list_eval_runs(limit))

    @app.post("/api/evals")
    def api_eval_create() -> Response:
        store: StateStore = app.config["STORE"]
        runner: JobRunner = app.config["RUNNER"]
        data = request.get_json(force=True)
        model_name = data.get("model_name", "").strip()
        model_id = data.get("model_id")
        eval_key = data.get("eval_key", "")
        sync_report = None
        if eval_key == "capability-gate":
            sync_report = capability_gate(model_name)
            report_path = PATHS.logs / f"capability_gate_{slugify(model_name)}.json"
            write_eval_report(report_path, sync_report)
            store.add_eval_run(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "eval_type": eval_key,
                    "status": "succeeded" if sync_report.get("ok") else "failed",
                    "report_path": str(report_path),
                    "metrics": sync_report,
                }
            )
            return jsonify({"sync": True, "report": sync_report}), 200
        elif eval_key == "tool-smoke-sync":
            sync_report = tool_smoke(model_name)
            report_path = PATHS.logs / f"tool_smoke_{slugify(model_name)}.json"
            write_eval_report(report_path, sync_report)
            store.add_eval_run(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "eval_type": eval_key,
                    "status": "succeeded" if sync_report.get("ok") else "failed",
                    "report_path": str(report_path),
                    "metrics": sync_report,
                }
            )
            return jsonify({"sync": True, "report": sync_report}), 200
        else:
            try:
                spec = get_eval(eval_key, model_name)
            except KeyError as exc:
                return jsonify({"error": str(exc)}), 400
            job_id = runner.start(
                kind=f"eval-{spec.key}",
                command=spec.command,
                cwd=PATHS.package_root,
                model_id=model_id,
                metadata={"eval": spec.key, "model_name": model_name},
            )
            store.add_eval_run(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "eval_type": spec.key,
                    "status": "running",
                }
            )
            return jsonify({"id": job_id, "kind": spec.key, "label": spec.label}), 201

    @app.get("/api/actions")
    def api_actions() -> Response:
        return jsonify(
            [
                {
                    "key": a.key,
                    "label": a.label,
                    "kind": a.kind,
                    "requires_model": a.requires_model,
                    "requires_dataset": a.requires_dataset,
                }
                for a in action_specs().values()
            ]
        )

    @app.get("/api/ollama/show")
    def api_ollama_show() -> Response:
        model = request.args.get("model", "")
        return jsonify(
            show_model(model, verbose=True, include_modelfile=True).to_dict()
        )

    # -----------------------------------------------------------------------
    # Compare API endpoints
    # -----------------------------------------------------------------------

    @app.get("/api/compare/models")
    def api_compare_models() -> Response:
        store: StateStore = app.config["STORE"]
        ids = request.args.getlist("id")
        all_models = store.list_models()
        selected = [m for m in all_models if str(m.get("id")) in ids]
        return jsonify(selected)

    @app.get("/api/compare/evals")
    def api_compare_evals() -> Response:
        store: StateStore = app.config["STORE"]
        ids = request.args.getlist("id")
        all_evals = store.list_eval_runs(200)
        selected = [e for e in all_evals if str(e.get("id")) in ids]
        return jsonify(selected)

    # -----------------------------------------------------------------------
    # Comparison views
    # -----------------------------------------------------------------------

    @app.get("/compare/models")
    def compare_models() -> str:
        store: StateStore = app.config["STORE"]
        all_models = store.list_models()
        selected_ids = set()
        for raw in request.args.getlist("id"):
            try:
                selected_ids.add(int(raw))
            except (ValueError, TypeError):
                pass
        section = request.args.get("section", "all")
        models = (
            [m for m in all_models if m.get("id") in selected_ids]
            if selected_ids
            else []
        )
        return render_template(
            "compare_models.html",
            all_models=all_models,
            models=models,
            selected_ids=selected_ids,
            section=section,
        )

    @app.route("/compare/evals", methods=["GET", "POST"])
    def compare_evals() -> str:
        store: StateStore = app.config["STORE"]
        selected = request.args.getlist("selected") or []
        if request.method == "POST":
            action = request.form.get("action", "")
            run_id = request.form.get("run_id")
            if action == "select" and run_id:
                selected = list(set(selected) | {run_id})
            elif action == "clear":
                selected = []
        selected_runs = []
        all_evals = store.list_eval_runs(200)
        for sid in selected:
            for run in all_evals:
                if str(run.get("id")) == sid:
                    selected_runs.append(run)
                    break
        metric_keys: list[str] = []
        if selected_runs:
            seen = set()
            for run in selected_runs:
                for k in (run.get("metrics") or {}).keys():
                    if k not in seen:
                        seen.add(k)
                        metric_keys.append(k)
        return render_template(
            "compare_evals.html",
            eval_runs=all_evals,
            selected=selected_runs,
            metric_keys=metric_keys,
        )

    return app


def _json_form(name: str) -> dict[str, Any]:
    raw = request.form.get(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        flash(f"{name} is not valid JSON; saved as empty object.", "error")
        return {}
    return value if isinstance(value, dict) else {}


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def _modelfile_spec_from_form() -> ModelfileSpec:
    params = {}
    for key in (
        "num_ctx",
        "num_predict",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repeat_penalty",
        "stop",
    ):
        value = request.form.get(f"param_{key}")
        if value not in (None, ""):
            params[key] = value
    return ModelfileSpec(
        from_ref=request.form.get("from_ref", "").strip(),
        renderer=request.form.get("renderer", DEFAULT_RENDERER).strip(),
        parser=request.form.get("parser", DEFAULT_PARSER).strip(),
        parameters=params,
        license_text=request.form.get("license_text") or None,
        template=request.form.get("template", "{{ .Prompt }}").strip() or None,
        requires=request.form.get("requires") or None,
    )


app = create_app()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Training Suite dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
