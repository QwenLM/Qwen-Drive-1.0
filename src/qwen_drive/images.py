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

"""Reading camera frames from a directory tree, a packed archive, or object storage.

A benchmark split references its frames by relative path. Those can live as ordinary
files, or be packed into Parquet shards by ``scripts/pack_images.py``, which turns tens
of thousands of small reads into a handful of memory-mapped ones. The archive stores the
original encoded bytes, so both routes decode to identical pixels.

"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

__all__ = ["ImageArchive"]

PATH_COLUMN = "path"
BYTES_COLUMN = "image"


class ImageArchive:
    """Random access to camera frames packed in Parquet shards."""

    def __init__(self, shards: list[str | Path]) -> None:
        import pyarrow.parquet as pq

        self._tables = []
        self._index: dict[str, tuple[int, int]] = {}
        for shard_index, shard in enumerate(shards):
            # Uncompressed shards let Arrow hand out slices of the mapped file instead of
            # copying every frame into memory.
            table = pq.read_table(str(shard), memory_map=True)
            self._tables.append(table)
            for row, path in enumerate(table.column(PATH_COLUMN).to_pylist()):
                self._index[path] = (shard_index, row)

    @classmethod
    def open(cls, root: str | Path) -> "ImageArchive":
        """Open every ``*.parquet`` shard in a directory, or a single shard file."""
        root = Path(root)
        shards = sorted(root.glob("*.parquet")) if root.is_dir() else [root]
        if not shards:
            raise FileNotFoundError(f"no parquet shards found at {root}")
        return cls(shards)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, path: str) -> bool:
        return path in self._index

    def read(self, path: str) -> Image.Image:
        """Decode one frame."""
        try:
            shard_index, row = self._index[path]
        except KeyError:
            raise KeyError(f"{path!r} is not in the archive ({len(self)} frames)") from None
        data = self._tables[shard_index].column(BYTES_COLUMN)[row].as_py()
        return Image.open(BytesIO(data)).convert("RGB")
