"""Qualify the full model plus learned pilot projector in unmodified Ollama.

Registers only an explicitly experimental local name and performs no registry
publication. Inference uses the public broker, automatic offload, and hard GPU
placement. Learned image quality is deliberately not claimed by this IO check.
"""
from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import urllib.error
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT.parent / "kolibri-1-20261004"
TAG = "kolibri-vision-training-pilot:q4_k_m-20261006"
GPU = "GPU-fc9d264d-8a20-bf8c-eb72-72a6dcbdaf2a"
MAIN_SHA = "fa6ccc6467702ea435876ed0e13f135ec268be8d7818ba66201de44eadb8181f"


def api(port, path, value=None, headers=None, timeout=300):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        data=json.dumps(value).encode() if value is not None else None,
        headers={"Content-Type":"application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def import_projector(path, digest):
    url = f"http://127.0.0.1:11436/api/blobs/sha256:{digest}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url,method="HEAD"),timeout=20): return
    except urllib.error.HTTPError as error:
        if error.code != 404: raise
    with path.open("rb") as file:
        request=urllib.request.Request(url,data=file,method="POST",headers={"Content-Length":str(path.stat().st_size)})
        with urllib.request.urlopen(request,timeout=120) as response:
            if response.status != 201: raise RuntimeError("Projector import failed")


def main():
    proof=json.loads((ROOT/"compatibility/pilot-native-projection-proof.json").read_text())
    if not proof["passed"]: raise RuntimeError("Native projection export has not passed")
    discovery=subprocess.run(["docker","gpu","discover"],capture_output=True,text=True,check=True)
    (ROOT/"compatibility/gpu-discovery-before-full-stock-pilot.json").write_text(discovery.stdout)
    observed=json.loads(discovery.stdout)
    print(json.dumps({"warnings":observed["warnings"],"active_leases":observed["active_leases"]}),flush=True)
    if any(GPU in lease["gpu_uuids"] for lease in observed["active_leases"]):
        raise RuntimeError("Required inference GPU still has an external lease")
    candidate=ROOT/"Kolibri-1-vision-pilot-mmproj.gguf"
    report=json.loads(candidate.with_suffix(".export.json").read_text())
    with candidate.open("rb") as file: digest=hashlib.file_digest(file,"sha256").hexdigest()
    if digest != report["sha256"]: raise RuntimeError("Candidate digest changed")
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:11436/api/blobs/sha256:{MAIN_SHA}",method="HEAD"),timeout=20): pass
    import_projector(candidate,digest)
    create={"model":TAG,"files":{"Kolibri-1-stock-Q4_K_M.gguf":"sha256:"+MAIN_SHA,
        candidate.name:"sha256:"+digest},"stream":False,"requires":"0.35.0",
        "template":(RELEASE/"kolibri-ollama-tools.gotmpl").read_text(),
        "info":{"capabilities":["completion","vision","tools","thinking"]},
        "parameters":{"num_ctx":4096,"num_predict":64,"temperature":0,"stop":["<|im_end|>"]}}
    if api(11436,"/api/create",create).get("status")!="success":raise RuntimeError("Full stock candidate creation failed")
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        sock.settimeout(30);sock.connect("/run/ollama-unify/gpu-negotiator.sock")
        sock.sendall((json.dumps({"action":"set_model_gpus","model":TAG,"gpu_uuids":[GPU]})+"\n").encode())
        result=json.loads(sock.makefile().readline())
        if not result.get("ok"):raise RuntimeError("Exact candidate placement failed")
    shown=api(11434,"/api/show",{"model":TAG})
    if not {"completion","vision","tools","thinking"}.issubset(shown["capabilities"]):raise RuntimeError("Candidate capability packing incomplete")
    examples=[json.loads(line) for line in (ROOT/"data-pilot-counted/examples.jsonl").read_text().splitlines()]
    example=next(e for e in examples if e["config"]=="docvqa" and e["pilot_split"]=="development_holdout")
    png=ROOT/"compatibility/pilot-stock-holdout-document.png"
    with Image.open(ROOT/"data-pilot-counted"/example["image"]) as image:
        image.convert("RGB").save(png)
    if png.stat().st_size>16*2**20:raise RuntimeError("Bounded test image exceeds limit")
    request={"model":TAG,"messages":[{"role":"user","content":example["user"],
        "images":[base64.b64encode(png.read_bytes()).decode()]}],"think":False,"stream":False,
        "keep_alive":"5m","options":{"num_gpu":-1,"num_ctx":4096,"num_predict":64,"temperature":0}}
    response=api(11434,"/api/chat",request,headers={"X-Ollama-Unify-GPU-UUIDs":GPU,
        "X-Ollama-Unify-Workload-Class":"background"},timeout=600)
    if not response.get("done") or response.get("prompt_eval_count",0)<100:
        raise RuntimeError("Full stock image input did not complete")
    with (ROOT/"compatibility/pilot-stock-cli.stdout").open("w") as stdout, (ROOT/"compatibility/pilot-stock-cli.stderr").open("w") as stderr:
        command=["ollama","run",TAG,"--think=false",example["user"]+" "+str(png)]
        cli=subprocess.run(command,env={**os.environ,"OLLAMA_HOST":"http://127.0.0.1:11434"},stdout=stdout,stderr=stderr,timeout=600)
    passed=cli.returncode==0
    receipt={"utc":datetime.now(timezone.utc).isoformat(),"passed":passed,
        "scope":"full released Kolibri Q4 weights plus learned pilot projector, stock API image request and ordinary CLI; OCR capability unqualified",
        "model":TAG,"capabilities":shown["capabilities"],"api_response":response,
        "cli_command":command,"cli_returncode":cli.returncode,"main_sha256":MAIN_SHA,"projector_sha256":digest,
        "optimizer_updates":proof["optimizer_updates"],"development_holdout_image":example["id"],
        "runtime_adapter_required":False,"published":False}
    (ROOT/"compatibility/full-stock-pilot-proof.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"passed":passed,"model":TAG,"capabilities":shown["capabilities"],
        "api_prompt_tokens":response.get("prompt_eval_count"),"cli_returncode":cli.returncode,"published":False}))
    return 0 if passed else 1


if __name__=="__main__":raise SystemExit(main())
