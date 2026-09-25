from __future__ import annotations

import os
import shlex
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import PATHS, PROJECT_ROOT, slugify, utc_now
from .state import StateStore


@dataclass(frozen=True)
class GpuLeaseSpec:
    owner: str
    justification: str
    expected_duration: int
    gpu_uuids: tuple[str, ...]
    ready_command: str
    vram_mib: int | None = None

    def validate(self) -> None:
        if not self.owner.strip():
            raise ValueError("GPU lease owner is required")
        if len(self.justification.strip()) < 12:
            raise ValueError("GPU lease justification must be specific")
        if self.expected_duration <= 0:
            raise ValueError("GPU lease expected_duration must be positive")
        if not self.gpu_uuids or any(not value.startswith("GPU-") for value in self.gpu_uuids):
            raise ValueError("GPU lease requires one or more GPU UUIDs")
        if not self.ready_command.strip():
            raise ValueError("GPU lease ready_command must prove model residency")
        if self.vram_mib is not None and self.vram_mib <= 0:
            raise ValueError("GPU lease vram_mib must be positive")


class JobRunner:
    """Run long operations in background subprocesses with persistent logs."""

    def __init__(self, store: StateStore | None = None) -> None:
        self.store = store or StateStore()
        self._processes: dict[int, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def start(
        self,
        *,
        kind: str,
        command: Iterable[str],
        cwd: Path | None = None,
        model_id: int | None = None,
        dataset_id: int | None = None,
        metadata: dict | None = None,
        environment: dict[str, str] | None = None,
        gpu_lease: GpuLeaseSpec | None = None,
    ) -> int:
        PATHS.ensure()
        command_list = [str(part) for part in command]
        process_environment = os.environ.copy()
        process_environment.update(environment or {})
        ready_file: Path | None = None
        if gpu_lease is not None:
            gpu_lease.validate()
            discovery = subprocess.run(
                ["docker", "gpu", "discover"],
                cwd=str(cwd or PROJECT_ROOT),
                text=True,
                capture_output=True,
                check=False,
            )
            if discovery.returncode != 0:
                raise RuntimeError(
                    "GPU broker discovery failed: "
                    + (discovery.stderr.strip() or discovery.stdout.strip() or "unknown error")
                )
            ready_command = gpu_lease.ready_command
            if ready_command == "auto":
                ready_dir = PATHS.state / "ready"
                ready_dir.mkdir(parents=True, exist_ok=True)
                ready_file = ready_dir / f"{slugify(gpu_lease.owner)}-{uuid.uuid4().hex}.json"
                process_environment["DISTILL_READY_FILE"] = str(ready_file)
                ready_command = f"test -s {shlex.quote(str(ready_file))}"
            wrapper = [
                "docker", "gpu", "run",
                "--owner", gpu_lease.owner,
                "--justification", gpu_lease.justification,
                "--expected-duration", str(gpu_lease.expected_duration),
            ]
            if gpu_lease.vram_mib is not None:
                wrapper.extend(["--vram-mib", str(gpu_lease.vram_mib)])
            for gpu_uuid in gpu_lease.gpu_uuids:
                wrapper.extend(["--gpu", gpu_uuid])
            # The installed negotiator uses argparse.REMAINDER; inserting a
            # literal `--` leaves it in args.command and makes Popen try to
            # execute a program named `--`.
            wrapper.extend(["--ready-command", ready_command])
            command_list = [*wrapper, *command_list]
            visible = ",".join(gpu_lease.gpu_uuids)
            process_environment["CUDA_VISIBLE_DEVICES"] = visible
            process_environment["DISTILL_GPUS"] = visible
        log_name = f"{utc_now().replace(':', '').replace('+', 'z')}-{slugify(kind)}.log"
        log_path = PATHS.logs / "jobs" / log_name
        job_id = self.store.create_job(
            kind=kind,
            command=command_list,
            cwd=cwd or PROJECT_ROOT,
            log_path=log_path,
            model_id=model_id,
            dataset_id=dataset_id,
            metadata={
                **(metadata or {}),
                **(
                    {
                        "gpu_lease": {
                            "owner": gpu_lease.owner,
                            "justification": gpu_lease.justification,
                            "expected_duration": gpu_lease.expected_duration,
                            "gpu_uuids": list(gpu_lease.gpu_uuids),
                            "vram_mib": gpu_lease.vram_mib,
                            "ready_command": gpu_lease.ready_command,
                            "ready_file": str(ready_file) if ready_file is not None else None,
                        }
                    }
                    if gpu_lease is not None
                    else {}
                ),
                "environment_keys": sorted((environment or {}).keys()),
            },
        )
        thread = threading.Thread(
            target=self._run,
            args=(job_id, command_list, cwd or PROJECT_ROOT, log_path, process_environment, ready_file),
            daemon=True,
        )
        thread.start()
        return job_id

    def _run(
        self,
        job_id: int,
        command: list[str],
        cwd: Path,
        log_path: Path,
        environment: dict[str, str],
        ready_file: Path | None,
    ) -> None:
        self.store.update_job(job_id, status="running", started_at=utc_now())
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                log.write(f"$ {' '.join(command)}\n\n")
                log.flush()
                proc = subprocess.Popen(
                    command,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=environment,
                )
                with self._lock:
                    self._processes[job_id] = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                returncode = proc.wait()
            status = "succeeded" if returncode == 0 else "failed"
            self.store.update_job(
                job_id,
                status=status,
                returncode=returncode,
                finished_at=utc_now(),
            )
        except Exception as exc:
            with log_path.open("a", encoding="utf-8", errors="replace") as log:
                log.write(f"\n[job-runner] {type(exc).__name__}: {exc}\n")
            self.store.update_job(
                job_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=utc_now(),
            )
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
            if ready_file is not None:
                ready_file.unlink(missing_ok=True)

    def cancel(self, job_id: int) -> bool:
        with self._lock:
            proc = self._processes.get(job_id)
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        self.store.update_job(job_id, status="cancelling")
        return True

    def read_log(self, job_id: int, max_chars: int = 200_000) -> str:
        job = self.store.get_job(job_id)
        if not job:
            return ""
        path = Path(job["log_path"])
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]
