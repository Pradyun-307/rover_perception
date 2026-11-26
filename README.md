# rover_perception

YOLO26n object detector for `aruco`, `hammer` and `mallet`, trained on Unity renders.

The data and trained weights are **not** in this repo. You build the training set from your own copy of `sequences/`.

Run all commands from the repo root.

## 1. Put the data in place

Copy your `sequences/` folder into the repo root:

```
rover_perception/
  sequences/
    traj_s0001/unity/rgb/*.jpg
    traj_s0001/unity/labels/*.txt
    traj_s0002/unity/...
    traj_s0003/unity/...
```

Each image `rgb/NNNNNN.jpg` needs a label `labels/NNNNNN.txt` in YOLO format.
Only the `unity/` folders of `traj_s0001`, `traj_s0002` and `traj_s0003` are used.

## 2. Split into train / val / test

Save the code below as `scripts/build_yolo_dataset.py` (or ask Claude Code: *"create the script from the README and run it"*).
Use it unchanged, so that everyone gets the same split.

<details>
<summary><code>scripts/build_yolo_dataset.py</code> (click to expand)</summary>

```python
#!/usr/bin/env python3
"""Build a YOLO train/val/test dataset from the Unity trajectory renders.

Source:  sequences/<traj>/unity/{rgb/*.jpg, labels/*.txt}  (labels already YOLO format)
Output:  YOLO_train+val/{images,labels}/{train,val,test}/<traj>_<frame>.{jpg,txt}
         YOLO_train+val/train.yaml
         YOLO_train+val/split_manifest.csv

Split: each trajectory is cut into contiguous temporal blocks which are assigned
to train/val/test (70/15/15 by block, seeded). Frames within GUARD of a boundary
between blocks of different splits are dropped, so near-duplicate consecutive
frames (24 fps) do not leak across splits.

Usage (from repo root):  python3 scripts/build_yolo_dataset.py
"""
import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

# ---------------- config ----------------
REPO = Path(__file__).resolve().parent.parent
SEQ_ROOT = REPO / "sequences"
SEQUENCES = ["traj_s0001", "traj_s0002", "traj_s0003"]
SRC_SUB = "unity"  # never x2
OUT = REPO / "YOLO_train+val"
BLOCK = 72    # frames per block (3 s @ 24 fps) -> 20 blocks per trajectory
N_VAL_BLOCKS = 3
N_TEST_BLOCKS = 3
GUARD = 6     # frames dropped each side of a split boundary
SEED = 0
NAMES = {0: "aruco", 1: "hammer", 2: "mallet"}
SPLITS = ("train", "val", "test")
# ----------------------------------------


def validate_label(path: Path) -> list[int]:
    """Return class ids in the label file; raise on malformed lines."""
    classes = []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{ln}: expected 5 fields, got {len(parts)}")
        cls = int(parts[0])
        if cls not in NAMES:
            raise ValueError(f"{path}:{ln}: unknown class {cls}")
        coords = [float(p) for p in parts[1:]]
        if not all(0.0 <= c <= 1.0 for c in coords):
            raise ValueError(f"{path}:{ln}: coords out of [0,1]: {coords}")
        classes.append(cls)
    return classes


def assign_splits(n_frames: int, rng: random.Random) -> list[str | None]:
    """Per-frame split label (None = dropped guard frame)."""
    n_blocks = (n_frames + BLOCK - 1) // BLOCK
    if n_blocks < N_VAL_BLOCKS + N_TEST_BLOCKS + 1:
        raise ValueError(f"only {n_blocks} blocks, too few to split")
    block_split = (["val"] * N_VAL_BLOCKS + ["test"] * N_TEST_BLOCKS
                   + ["train"] * (n_blocks - N_VAL_BLOCKS - N_TEST_BLOCKS))
    rng.shuffle(block_split)

    frame_split: list[str | None] = [block_split[i // BLOCK] for i in range(n_frames)]
    for b in range(1, n_blocks):
        if block_split[b] != block_split[b - 1]:
            edge = b * BLOCK
            for i in range(max(0, edge - GUARD), min(n_frames, edge + GUARD)):
                frame_split[i] = None
    return frame_split


def main() -> int:
    rng = random.Random(SEED)

    for split in SPLITS:
        for kind in ("images", "labels"):
            d = OUT / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    manifest = []
    img_count = Counter()
    box_count = {s: Counter() for s in SPLITS}
    traj_in_split = {s: set() for s in SPLITS}
    dropped = 0

    for traj in SEQUENCES:
        src = SEQ_ROOT / traj / SRC_SUB
        images = sorted((src / "rgb").glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"no images in {src / 'rgb'}")
        splits = assign_splits(len(images), rng)

        for img, split in zip(images, splits):
            lbl = src / "labels" / f"{img.stem}.txt"
            if not lbl.exists():
                raise FileNotFoundError(f"missing label for {img}")
            classes = validate_label(lbl)
            if split is None:
                dropped += 1
                manifest.append((traj, img.stem, "dropped_guard"))
                continue
            name = f"{traj}_{img.stem}"
            shutil.copy2(img, OUT / "images" / split / f"{name}.jpg")
            shutil.copy2(lbl, OUT / "labels" / split / f"{name}.txt")
            manifest.append((traj, img.stem, split))
            img_count[split] += 1
            box_count[split].update(classes)
            traj_in_split[split].add(traj)

    (OUT / "train.yaml").write_text(
        "# YOLO dataset config. `path` is rewritten by the training notebook.\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + "".join(f"  {k}: {v}\n" for k, v in NAMES.items())
    )
    with open(OUT / "split_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj", "frame", "split"])
        w.writerows(manifest)

    print(f"Dropped guard frames: {dropped}")
    for s in SPLITS:
        boxes = ", ".join(f"{NAMES[c]}={box_count[s][c]}" for c in NAMES)
        print(f"{s:5s}: {img_count[s]:5d} images | boxes: {boxes} | trajs: {sorted(traj_in_split[s])}")
    print(f"Wrote {OUT / 'train.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

</details>

Then run:

```
python3 scripts/build_yolo_dataset.py
```

On Windows use `python` instead of `python3`. The script needs only the standard library.

It writes:

- `YOLO_train+val/images/{train,val,test}/`
- `YOLO_train+val/labels/{train,val,test}/`
- `YOLO_train+val/split_manifest.csv`: which split each frame went to
- `YOLO_train+val/train.yaml`: dataset config (already in the repo; rewritten with the same content)

With the original data you should get **2850 train, 546 val, 540 test** images, with 384 frames dropped.
If your numbers differ, your copy of `sequences/` is not the same.

Files are renamed to `<traj>_<frame>.jpg` / `.txt` so names do not clash. Classes: `0 aruco`, `1 hammer`, `2 mallet`.

**How the split works:** each trajectory is cut into blocks of 72 consecutive frames (3 s), and whole blocks
are assigned to train/val/test (about 70/15/15) with a fixed seed. Frames within 6 of a boundary between
different splits are dropped, so near-identical neighbouring frames do not end up on both sides.

To change the sequences, block size or ratios, edit the config block at the top of the script. Re-running it
deletes and rebuilds `images/` and `labels/`. Keep the defaults if you want results comparable with the team's.

Val and test come from the same trajectories as train, so test scores will likely be optimistic compared
with a new trajectory or real camera footage.

## 3. Set up the environment

Linux:

```
unset PYTHONPATH    # needed if your shell sources ROS
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Then install PyTorch **with CUDA support** using the command for your OS from
<https://pytorch.org/get-started/locally/>, followed by:

```
pip install ultralytics ipykernel matplotlib pyyaml
python -c "import torch; print(torch.cuda.is_available())"
```

This must print `True`. If it prints `False`, you have a CPU-only build of PyTorch or a driver problem,
and training will be far too slow.

## 4. Train

Open `YOLO_train+val/train_yolo26n_local.ipynb` (for example in VS Code), pick `.venv` as the kernel and
run all cells.

Settings are in the first code cell:

| setting | default | notes |
|---|---|---|
| `IMGSZ` | 1280 | native image width; 640 is faster but loses small objects |
| `EPOCHS` | 100 | |
| `BATCH` | 2 | conservative; try 4 or 8, or `-1` (GPU only) to auto-size. Lower it on out-of-memory errors |
| `WORKERS` | 2 | raise if the GPU is waiting for data and you have spare RAM |
| `FRACTION` | 1.0 | use e.g. 0.02 with `EPOCHS = 1` for a quick trial; give it another `RUN_NAME` |
| `RESUME` | False | set `True` to continue an interrupted run |
| `FORCE_RETRAIN` | False | training is skipped if `best.pt` already exists for this `RUN_NAME` |

Outputs:

- `YOLO_train+val/runs/<RUN_NAME>/`: logs, plots and checkpoints
- `YOLO_train+val/weights/<RUN_NAME>_best.pt`: final model
