"""Sample actual image/answer examples from immutable public Parquet shards.

Range reads keep shards remote. Transfer, decode, output, row, and free-disk
limits apply at the point of use. This pilot is not an official benchmark.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

from huggingface_hub import HfFileSystem
from PIL import Image, ImageOps
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS = 32000000


class Budget:
    def __init__(self, transfer, artifacts, root):
        self.transfer_limit, self.output_limit, self.root = transfer, artifacts, root
        self.transferred = self.output = 0

    def reserve_transfer(self, count):
        if count < 0 or self.transferred + count > self.transfer_limit:
            raise RuntimeError("Pilot network budget exhausted before the next range read")
        self.transferred += count

    def write(self, path, data):
        if path.exists():
            return
        if self.output + len(data) > self.output_limit:
            raise RuntimeError("Pilot artifact budget exhausted before writing")
        if shutil.disk_usage(self.root).free - len(data) < 64 * 2**30:
            raise RuntimeError("Required shared-disk headroom is unavailable")
        path.write_bytes(data)
        self.output += len(data)


def bounded_file(fs, source, budget):
    file = fs.open(source, "rb", cache_type="none", block_size=65536)
    fetch = file._fetch_range
    def bounded_fetch(start, end):
        budget.reserve_transfer(end - start)
        return fetch(start, end)
    file._fetch_range = bounded_fetch
    # fsspec caches the original bound fetch method when constructing its
    # cache; replacing only file._fetch_range would leave reads uncounted.
    file.cache.fetcher = bounded_fetch
    return file


def image_identity(image):
    # Decoded hashes detect copies even when container compression differs.
    canonical = ImageOps.exif_transpose(image).convert("RGB")
    exact = hashlib.sha256(str(canonical.size).encode() + canonical.tobytes()).hexdigest()
    small = np.asarray(canonical.resize((9, 8), Image.Resampling.LANCZOS).convert("L"))
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    perceptual = f"{sum(int(v) << i for i,v in enumerate(bits)):016x}"
    return exact, perceptual, canonical.size


def accepted_turns(row, thresholds, counters):
    turns = row.get("texts", [])
    for index, turn in enumerate(turns):
        if not turn.get("user", "").strip() or not turn.get("assistant", "").strip():
            counters["empty_dialogue"] += 1
            continue
        ratings = {}
        for key, threshold in thresholds.items():
            values = row.get(key + "_ratings", [])
            if index >= len(values):
                counters["missing_quality_rating"] += 1
                break
            ratings[key] = values[index]
            if values[index] < threshold:
                counters["quality_" + key] += 1
                break
        else:
            yield turn, ratings


def sample(recipe, out, per_config, names, max_rows, transfer_limit, output_limit, pool="finevision"):
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    if (out / "examples.jsonl").exists():
        raise ValueError("Choose a fresh pilot output to preserve prior sampling evidence")
    source = recipe["sources"][pool]
    if pool == "finevision":
        configurations = {c["name"]: (group, c) for group, value in recipe["instruction_groups"].items()
                          for c in value["configurations"]}
    else:
        configurations = {name: ("alignment_caption", {"shards": [p for p in source["all_training_shards"] if p.startswith(name + "/")]})
                          for name in names}
        if any(not c["shards"] for _,c in configurations.values()):
            raise ValueError("Alignment source has no pinned shards")
    budget = Budget(transfer_limit, output_limit, out)
    fs = HfFileSystem()
    counts, rejects, errors, examples, seen_images = {}, Counter(), {}, [], set()
    thresholds = recipe["quality"]["per_turn_minimum"] if pool == "finevision" else {}
    started = time.monotonic()
    for name in names:
        group, config = configurations[name]
        shard = config["shards"][0]
        path = f"datasets/{source['repo']}@{source['revision']}/{shard}"
        accepted = rows = 0
        try:
            with bounded_file(fs, path, budget) as file:
                parquet = pq.ParquetFile(file)
                expected = {"images", "texts"} if pool == "finevision" else {"image", "caption"}
                if not expected.issubset(parquet.schema_arrow.names):
                    raise ValueError("Source schema is not a real image/dialogue dataset")
                for rowgroup in range(parquet.metadata.num_row_groups):
                    if rows >= max_rows or accepted >= per_config:
                        break
                    metadata = parquet.metadata.row_group(rowgroup)
                    compressed = sum(metadata.column(i).total_compressed_size for i in range(metadata.num_columns))
                    if compressed > 128 * 2**20:
                        raise ValueError("Row group exceeds the bounded pilot decode size")
                    table = parquet.read_row_group(rowgroup, use_threads=False)
                    for row in table.to_pylist():
                        if rows >= max_rows or accepted >= per_config:
                            break
                        source_row = rows
                        rows += 1
                        if pool == "onevision85m":
                            caption = row.get("caption")
                            if not isinstance(caption, str) or not 40 <= len(caption.strip()) <= 20000:
                                rejects["invalid_alignment_caption"] += 1
                                continue
                            row = {"images": [row["image"]], "texts": [{"user": "Describe this image in detail.", "assistant": caption}]}
                        images = row.get("images", [])
                        if len(images) != 1:
                            rejects["pilot_requires_one_image"] += 1
                            continue
                        raw = images[0].get("bytes") if isinstance(images[0], dict) else None
                        if not raw or len(raw) > 32 * 2**20:
                            rejects["missing_or_oversized_image_bytes"] += 1
                            continue
                        turns = list(accepted_turns(row, thresholds, rejects))
                        if not turns:
                            continue
                        try:
                            with Image.open(io.BytesIO(raw)) as image:
                                if image.width * image.height > 32000000:
                                    raise ValueError("Decoded image pixel limit exceeded")
                                image.load()
                                exact, perceptual, size = image_identity(image)
                        except Exception:
                            rejects["invalid_image"] += 1
                            continue
                        if exact in seen_images:
                            rejects["duplicate_image"] += 1
                            continue
                        seen_images.add(exact)
                        relative = Path("images") / (exact + ".image")
                        budget.write(out / relative, raw)
                        turn, ratings = turns[0]
                        example = {"id": exact + ":" + name, "group": group, "config": name,
                            "repo": source["repo"], "revision": source["revision"], "split": "train",
                            "shard": shard, "row": source_row, "image": str(relative), "size": size,
                            "image_sha256": exact, "image_dhash": perceptual, "ratings": ratings,
                            "user": turn["user"], "assistant": turn["assistant"],
                            "pilot_split": "development_holdout" if accepted % 5 == 4 else "training"}
                        examples.append(example)
                        accepted += 1
        except Exception as error:
            # Do not print signed CDN URLs or HTTP authorization details.
            errors[name] = type(error).__name__
        counts[name] = {"rows_read": rows, "accepted_images": accepted}
        print(json.dumps({"config": name, **counts[name], "transfer_bytes": budget.transferred,
            "error_type": errors.get(name)}), flush=True)
    raw = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples).encode()
    budget.write(out / "examples.jsonl", raw)
    report = {"utc": datetime.now(timezone.utc).isoformat(), "passed": all(counts[n]["accepted_images"] >= per_config for n in names),
        "scope": "real training-media pilot; separate official benchmark exclusion still required before scaling",
        "source": {k:v for k,v in source.items() if k != "all_training_shards"}, "counts": counts, "rejected_turns_or_rows": dict(rejects), "error_types": errors,
        "transfer_bytes_upper_bound": budget.transferred, "artifact_bytes_written": budget.output,
        "seconds": time.monotonic() - started, "accepted_examples": len(examples),
        "training_examples": sum(e["pilot_split"] == "training" for e in examples),
        "development_holdout_examples": sum(e["pilot_split"] != "training" for e in examples),
        "trained_examples": 0, "limits": {"transfer": transfer_limit, "artifacts": output_limit,
            "rows_per_config": max_rows, "pixels_per_image": 32000000}}
    (out / "sampling-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data-pilot")
    parser.add_argument("--configs", nargs="+", default=["docvqa", "textvqa", "screenqa", "chartqa", "sharegpt4v(coco)"])
    parser.add_argument("--pool", choices=["finevision", "onevision85m"], default="finevision")
    parser.add_argument("--per-config", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--max-transfer-bytes", type=int, default=2 * 2**30)
    parser.add_argument("--max-output-bytes", type=int, default=128 * 2**20)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT) or args.max_rows <= 0 or args.per_config <= 0:
        raise ValueError("Pilot must use a bounded run-local output and positive limits")
    recipe = json.loads((ROOT / "data-mixture.json").read_text())
    report = sample(recipe, args.out, args.per_config, args.configs, args.max_rows,
        args.max_transfer_bytes, args.max_output_bytes, args.pool)
    print(json.dumps({"passed": report["passed"], "accepted_examples": report["accepted_examples"],
        "training_examples": report["training_examples"], "development_holdout_examples": report["development_holdout_examples"]}))
    raise SystemExit(0 if report["passed"] else 1)
