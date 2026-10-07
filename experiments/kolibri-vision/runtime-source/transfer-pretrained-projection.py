"""Initialize a Kolibri bridge from existing Qwen vision and shared vocabulary.

This is a CPU-only ridge alignment, not a claim of preserved vision ability.
Independent vocabulary rows measure generalization of the linear mapping;
actual image evaluations are required before any release.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE / "training_suite/vendor/kolibri-port/llama.cpp/gguf-py"))
import gguf

DONOR = Path('/srv/ollama/models/blobs/sha256-852922174ee4f76621df26105333f1dfe2171cdfb60ebe5a4b013836681a8a77')
VISION = Path('/srv/ollama/models/blobs/sha256-2635637dd72b4d39a97331373a73366b69a0a6a14151b2a1da0a97b311e2461f')
TARGET = ROOT.parent / 'kolibri-1-20261004/Kolibri-1-stock-BF16-release.gguf'


def tensors(reader):
    return {t.name: t for t in reader.tensors}


def dense(tensor, rows=None):
    raw = tensor.data if rows is None else tensor.data[rows]
    value = gguf.quants.dequantize(raw, tensor.tensor_type)
    return torch.from_numpy(np.array(value, dtype=np.float32, copy=True))


def metrics(actual, expected, mean):
    residual = (actual - expected).square().sum()
    variance = (expected - mean).square().sum()
    return {'r_squared': float(1 - residual / variance),
        'rmse': float((actual - expected).square().mean().sqrt()),
        'cosine_mean': float(torch.nn.functional.cosine_similarity(actual, expected).mean()),
        'target_rms': float(expected.square().mean().sqrt()),
        'predicted_rms': float(actual.square().mean().sqrt())}


def main():
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    donor, target, vision = [gguf.GGUFReader(p) for p in (DONOR, TARGET, VISION)]
    source_tokens = donor.fields['tokenizer.ggml.tokens'].contents()
    target_tokens = target.fields['tokenizer.ggml.tokens'].contents()
    source_types = donor.fields['tokenizer.ggml.token_type'].contents()
    target_types = target.fields['tokenizer.ggml.token_type'].contents()
    source_ids = {token: i for i, token in enumerate(source_tokens) if source_types[i] == 1}
    shared = [(source_ids[token], i) for i, token in enumerate(target_tokens)
        if target_types[i] == 1 and token in source_ids and len(token) >= 2]
    rng = np.random.default_rng(6100626)
    rng.shuffle(shared)
    shared = shared[:40000]
    train_count = min(32000, int(len(shared) * .8))
    pairs = np.asarray(shared, dtype=np.int64)
    print(json.dumps({'shared_tokens_selected': len(shared), 'training_rows': train_count,
        'validation_rows': len(shared)-train_count, 'cuda_used': False}), flush=True)
    x = dense(tensors(donor)['token_embd.weight'], pairs[:,0])
    y = dense(tensors(target)['token_embd.weight'], pairs[:,1])
    xm, ym = x[:train_count].mean(0), y[:train_count].mean(0)
    xc, yc = x[:train_count]-xm, y[:train_count]-ym
    gram = xc.T @ xc / train_count
    cross = xc.T @ yc / train_count
    diagonal_scale = float(gram.diagonal().mean())
    fits = []
    best = None
    for factor in [.01, .1, 1.0]:
        regularized = gram.clone()
        regularized.diagonal().add_(factor * diagonal_scale)
        mapping = torch.cholesky_solve(cross, torch.linalg.cholesky(regularized))
        predicted = (x[train_count:] - xm) @ mapping + ym
        result = {'ridge_factor': factor, **metrics(predicted, y[train_count:], ym)}
        fits.append(result)
        print(json.dumps(result), flush=True)
        if best is None or result['r_squared'] > best[0]['r_squared']:
            best = (result, mapping)
    result, mapping = best
    vt = tensors(vision)
    source_weight = dense(vt['mm.2.weight']).reshape(4096,4608)
    source_bias = dense(vt['mm.2.bias']).reshape(4096)
    weight = mapping.T @ source_weight
    bias = (source_bias-xm) @ mapping + ym
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise RuntimeError('Nonfinite projection transfer')
    # Check the composition using unseen random features, independently of GGUF export.
    torch.manual_seed(6100626)
    features = torch.randn(13,4608) * .07
    expected = (torch.nn.functional.linear(features,source_weight,source_bias)-xm) @ mapping + ym
    actual = torch.nn.functional.linear(features,weight,bias)
    error = float((expected-actual).square().mean().sqrt())
    if error > 1e-6:
        raise RuntimeError('Projection composition failed')
    checkpoint = ROOT / 'transferred-projector.pt'
    if checkpoint.exists():
        raise RuntimeError('Refusing to overwrite existing transfer candidate')
    report = {'utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'pretrained projector initialization through shared-token ridge alignment; vision quality unqualified',
        'donor_language_sha256': DONOR.name.removeprefix('sha256-'),
        'donor_projector_sha256': VISION.name.removeprefix('sha256-'),
        'target_language_sha256': '1e029937d835d4dd8c2a1ee775464585403b7492b7f5e987e19b231d679fc634',
        'shared_tokens': len(shared), 'training_rows': train_count,
        'heldout_vocabulary_rows': len(shared)-train_count,
        'selected': result, 'candidates': fits, 'composition_rmse': error,
        'seconds': time.monotonic()-started, 'language_weights_changed': False,
        'training_examples_used': 0, 'published': False}
    torch.save({'state_dict': {'weight': weight, 'bias': bias}, 'optimizer_updates': 0,
        'pretrained_transfer': report, 'scope': report['scope']}, checkpoint)
    report['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (ROOT/'compatibility/pretrained-projection-transfer.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
