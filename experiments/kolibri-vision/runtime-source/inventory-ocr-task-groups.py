"""Read bounded OCRBench task metadata without downloading image columns."""
from collections import Counter
import importlib.util
import json
from pathlib import Path

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sampling', ROOT/'runtime-source/sample-training-data.py')
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)


def main():
    info = json.loads((ROOT/'data-sources/lmms-lab--OCRBench-v2-benchmark-info.json').read_text())
    budget = sampling.Budget(64*2**20, 2*2**20, ROOT)
    fs = HfFileSystem()
    groups = []
    for entry in info['siblings']:
        shard = entry['rfilename']
        if not shard.endswith('.parquet'):
            continue
        with sampling.bounded_file(fs, f"datasets/{info['id']}@{info['sha']}/{shard}", budget) as file:
            parquet = pq.ParquetFile(file)
            for index in range(parquet.metadata.num_row_groups):
                meta = parquet.metadata.row_group(index)
                rows = parquet.read_row_group(index, columns=['type', 'answers'], use_threads=False).to_pylist()
                types = Counter(row['type'] for row in rows)
                short = Counter()
                for row in rows:
                    answers = row['answers']
                    if isinstance(answers, list) and answers and all(
                        isinstance(answer, str) and 1 <= len(answer) <= 80 and
                        not any(c in answer for c in '[]{}') for answer in answers):
                        short[row['type']] += 1
                image_bytes = sum(meta.column(i).total_compressed_size for i in range(meta.num_columns)
                                  if meta.column(i).path_in_schema.startswith('image.'))
                group = {'shard': shard, 'rowgroup': index, 'rows': meta.num_rows,
                         'types': dict(types), 'short_answer_types': dict(short),
                         'image_column_compressed_bytes': image_bytes}
                groups.append(group)
                print(json.dumps(group), flush=True)
    report = {'repo': info['id'], 'revision': info['sha'], 'groups': groups,
              'transfer_bytes_upper_bound': budget.transferred, 'transfer_limit': budget.transfer_limit}
    sampling.Budget.write(budget, ROOT/'compatibility/ocrbench-task-group-inventory.json',
                          (json.dumps(report, indent=2)+'\n').encode())


if __name__ == '__main__':
    main()
