#!/usr/bin/env python
# Copyright 2026 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pack the frames a scene file references into memory-mappable Parquet shards.

A split needs twelve small image reads per scene, which dominates wall clock on network
storage. Packing them into a few uncompressed shards turns that into memory-mapped
reads of already-encoded bytes, so the decoded pixels are unchanged.

    python scripts/pack_images.py \
        --scenes scenes/waymo_e2e_val.jsonl \
        --image-root /data/waymo_e2e/extracted/images \
        --output archives/waymo_e2e_val

Pass the result to the eval scripts as ``--image-archive archives/waymo_e2e_val``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from qwen_drive.images import BYTES_COLUMN, PATH_COLUMN

SCHEMA = pa.schema([(PATH_COLUMN, pa.string()), (BYTES_COLUMN, pa.binary())])


def referenced_paths(scenes: Path, limit: int | None) -> list[str]:
    """Every distinct frame path in a scene file, in first-seen order."""
    seen: dict[str, None] = {}
    with open(scenes) as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            for item in json.loads(line)["messages"][0]["content"]:
                if "image" in item:
                    seen.setdefault(item["image"], None)
    return list(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="destination directory")
    parser.add_argument("--shard-mib", type=float, default=512.0)
    parser.add_argument("--limit", type=int, default=None, help="only the first N scenes")
    args = parser.parse_args()

    paths = referenced_paths(args.scenes, args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    limit_bytes = int(args.shard_mib * 1024**2)

    shard_paths: list[str] = []
    shard_blobs: list[bytes] = []
    shard_bytes = 0
    written = 0

    def flush() -> None:
        nonlocal shard_paths, shard_blobs, shard_bytes, written
        if not shard_paths:
            return
        target = args.output / f"images-{written:05d}.parquet"
        # No compression: the frames are already JPEG, and uncompressed shards can be
        # memory-mapped instead of decompressed into memory on open.
        pq.write_table(
            pa.table({PATH_COLUMN: shard_paths, BYTES_COLUMN: shard_blobs}, schema=SCHEMA),
            target,
            compression="none",
        )
        print(f"wrote {target.name} ({len(shard_paths)} frames, {shard_bytes / 1024**2:.0f} MiB)")
        shard_paths, shard_blobs, shard_bytes = [], [], 0
        written += 1

    for relative in tqdm(paths, desc="packing"):
        data = (args.image_root / relative).read_bytes()
        shard_paths.append(relative)
        shard_blobs.append(data)
        shard_bytes += len(data)
        if shard_bytes >= limit_bytes:
            flush()
    flush()
    print(f"packed {len(paths)} frames into {written} shard(s) under {args.output}")


if __name__ == "__main__":
    main()
