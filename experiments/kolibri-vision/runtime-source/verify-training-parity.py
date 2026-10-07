"""Verify decoder logits and projector-input gradients against stock Ollama."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("decoder", Path(__file__).with_name("frozen-decoder.py"))
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


def main():
    torch.set_num_threads(4)
    folder = ROOT / "compatibility/head128"
    proof = folder / "stock-cpu-interface-proof.json"
    if not json.loads(proof.read_text())["passed"]:
        raise RuntimeError("Stock CPU interface proof is incomplete")
    model = decoder.FrozenAFMoE(folder / "afmoe-interface-fixture.gguf", checkpoint_layers=True)
    reports = []
    for case in range(3):
        chunks = json.loads(Path(str(proof) + f".case{case}.inputs.json").read_text())["chunks"]
        embeddings, images = [], []
        for chunk in chunks:
            if chunk["type"] == "text":
                embeddings.append(model.embeddings(torch.tensor(chunk["tokens"], dtype=torch.long)))
            else:
                image = torch.tensor(chunk["embeddings"], dtype=torch.float32).reshape(-1, model.hidden)
                image.requires_grad_(True)
                images.append(image)
                embeddings.append(image)
        logits = model(torch.cat(embeddings), output_positions=[-1])[0]
        expected = np.fromfile(str(proof) + f".case{case}.logits.bin", dtype=np.float32)
        actual = logits.detach().numpy()
        rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
        error = float(np.abs(actual - expected).max())
        torch.nn.functional.cross_entropy(logits[None], torch.tensor([int(expected.argmax())])).backward()
        gradients = [float(x.grad.norm()) for x in images]
        finite = all(bool(torch.isfinite(x.grad).all()) for x in images)
        reports.append({"case": case, "logit_rmse": rmse, "logit_max_error": error,
                        "top1_matches": int(actual.argmax()) == int(expected.argmax()),
                        "image_gradient_norms": gradients, "finite_gradients": finite,
                        "passed": rmse < 2e-4 and error < 2e-3 and finite and all(g > 0 for g in gradients)})
    # Qualify the GPU-oriented grouped implementation independently as well.
    grouped = decoder.FrozenAFMoE(folder / "afmoe-interface-fixture.gguf", checkpoint_layers=False, grouped_experts=True)
    with torch.no_grad():
        grouped_logits = grouped(torch.cat([e.detach() for e in embeddings]), output_positions=[-1])[0]
    grouped_rmse = float(torch.mean((grouped_logits - logits.detach()).square()).sqrt())
    result = {"passed": all(r["passed"] for r in reports) and grouped_rmse < 2e-4,
              "scope": "tiny F32 decoder parity, image gradients, grouped routing; full BF16 training still unqualified",
              "cases": reports, "grouped_expert_logit_rmse": grouped_rmse}
    (folder / "training-decoder-parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise RuntimeError("Training decoder parity failed; do not start training")


if __name__ == "__main__":
    main()
