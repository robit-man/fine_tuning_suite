from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Iterable

from .config import PATHS, PROJECT_ROOT, slugify, utc_now
from .state import StateStore


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
    ) -> int:
        PATHS.ensure()
        command_list = [str(part) for part in command]
        log_name = f"{utc_now().replace(':', '').replace('+', 'z')}-{slugify(kind)}.log"
        log_path = PATHS.logs / "jobs" / log_name
        job_id = self.store.create_job(
            kind=kind,
            command=command_list,
            cwd=cwd or PROJECT_ROOT,
            log_path=log_path,
            model_id=model_id,
            dataset_id=dataset_id,
            metadata=metadata or {},
        )
        thread = threading.Thread(
            target=self._run,
            args=(job_id, command_list, cwd or PROJECT_ROOT, log_path),
            daemon=True,
        )
        thread.start()
        return job_id

    def _run(self, job_id: int, command: list[str], cwd: Path, log_path: Path) -> None:
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
