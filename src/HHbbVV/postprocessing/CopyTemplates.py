from __future__ import annotations

import shutil
from pathlib import Path


def copy_missing_directories(src_dir, dest_dir):
    # Get all directories in source
    src_dirs = [d for d in src_dir.iterdir() if d.is_dir()]

    # Get existing directories in destination
    dest_dirs = [d.name for d in dest_dir.iterdir() if d.is_dir()]

    # Copy missing directories
    for src in src_dirs:
        if src.name not in dest_dirs:
            print(f"Copying {src.name}")
            shutil.copytree(src, dest_dir / src.name)
        else:
            print(f"Skipping {src.name} - already exists")


if __name__ == "__main__":
    src_dir = Path("templates/25Feb9AllRes")
    dest_dir = Path("/ceph/cms/store/user/rkansal/bbVV/templates/25Feb8XHYFix/")
    copy_missing_directories(src_dir, dest_dir)
