"""
兼容Unitree G1的官方资产包装器。
"""

from pathlib import Path
from typing import Any

def update_assets(
    assets: dict[str, Any],
    path: str | Path,
    meshdir: str | None = None,
    glob: str = "*",
    recursive: bool = False,
):
    for f in Path(path).glob(glob):
        if f.is_file():
            asset_key = f"{meshdir}/{f.name}" if meshdir else f.name
            assets[asset_key] = f.read_bytes()
        elif f.is_dir() and recursive:
            update_assets(assets, f, meshdir, glob, recursive)