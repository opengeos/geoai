#!/usr/bin/env python3
"""Preprocess DALES LAS tiles into Pointcept-compatible .npy blocks.

Reads DALES LAS files (29 train + 11 test), splits each ~500x500 m tile
into non-overlapping spatial blocks, extracts coordinates / intensity /
labels, and saves in Pointcept's expected .npy format.

Usage:
    python preprocess_dales_ptv3.py \\
        --input_dir dales_las \\
        --output_dir data/dales_ptv3 \\
        --block_size 50 \\
        --val_ratio 0.15

Expected input layout:
    dales_las/
    ├── train/   (29 .las tiles)
    └── test/    (11 .las tiles)

Output layout:
    data/dales_ptv3/
    ├── train/
    │   ├── <tile>_b00_00/
    │   │   ├── coord.npy      (N, 3)  float32  centred XYZ
    │   │   ├── strength.npy   (N, 1)  float32  normalised return_number
    │   │   └── segment.npy    (N,)    int32    class labels 0-8
    │   └── ...
    ├── val/     (auto-split from train)
    ├── test/
    └── metadata.json   (class names, weights, split info)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import laspy
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

DALES_CLASSES: dict[int, str] = {
    0: "unknown",
    1: "ground",
    2: "vegetation",
    3: "cars",
    4: "trucks",
    5: "power_lines",
    6: "fences",
    7: "poles",
    8: "buildings",
}
NUM_CLASSES = 9
IGNORE_INDEX = 0


# ── I/O helpers ───────────────────────────────────────────────────────


def read_tile(
    las_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a DALES LAS tile.

    Returns
    -------
    coord : (N, 3) float32
    strength : (N, 1) float32  normalised return_number [0, 1] (falls back to intensity)
    segment : (N,) int32  class labels clipped to [0, 8]
    """
    las = laspy.read(str(las_path))

    coord = np.column_stack([las.x, las.y, las.z]).astype(np.float32)

    # DALES tiles have zero intensity but useful multi-return info.
    # Use return_number (normalised to [0, 1]) as the feature channel;
    # fall back to intensity if return_number is absent.
    if hasattr(las, "return_number"):
        rn = np.array(las.return_number, dtype=np.float32)
        peak = rn.max()
        strength = (rn / peak if peak > 0 else rn).reshape(-1, 1)
    elif hasattr(las, "intensity"):
        intensity = np.array(las.intensity, dtype=np.float32)
        peak = intensity.max()
        strength = (intensity / peak if peak > 0 else intensity).reshape(-1, 1)
    else:
        strength = np.zeros((len(coord), 1), dtype=np.float32)

    segment = np.array(las.classification, dtype=np.int32)
    segment = np.clip(segment, 0, NUM_CLASSES - 1)

    return coord, strength, segment


def split_into_blocks(
    coord: np.ndarray,
    strength: np.ndarray,
    segment: np.ndarray,
    block_size: float = 50.0,
    min_points: int = 100,
) -> list[dict[str, np.ndarray | str | int]]:
    """Split a tile into non-overlapping XY blocks.

    Each block's coordinates are centred (mean-subtracted) so that the
    model receives values near zero regardless of UTM offsets.
    """
    x_min, y_min = coord[:, 0].min(), coord[:, 1].min()
    x_max, y_max = coord[:, 0].max(), coord[:, 1].max()

    blocks: list[dict] = []
    bx = 0
    x = x_min
    while x < x_max:
        by = 0
        y = y_min
        while y < y_max:
            mask = (
                (coord[:, 0] >= x)
                & (coord[:, 0] < x + block_size)
                & (coord[:, 1] >= y)
                & (coord[:, 1] < y + block_size)
            )
            n = int(mask.sum())
            if n >= min_points:
                block_coord = coord[mask].copy()
                # Centre in float64 to avoid catastrophic cancellation
                # with large UTM offsets (Y ~ 5 × 10⁶ causes f32 precision loss).
                centre = block_coord.astype(np.float64).mean(axis=0)
                centred = (block_coord.astype(np.float64) - centre).astype(np.float32)
                blocks.append(
                    {
                        "coord": centred,
                        "strength": strength[mask].copy(),
                        "segment": segment[mask].copy(),
                        "grid_idx": f"b{bx:02d}_{by:02d}",
                        "n_points": n,
                    }
                )
            y += block_size
            by += 1
        x += block_size
        bx += 1

    return blocks


def save_block(block: dict, output_dir: Path) -> None:
    """Persist a single block as three .npy files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "coord.npy", block["coord"])
    np.save(output_dir / "strength.npy", block["strength"])
    np.save(output_dir / "segment.npy", block["segment"])


# ── Processing ────────────────────────────────────────────────────────


def process_tiles(
    las_paths: list[Path],
    output_dir: Path,
    block_size: float,
    min_points: int,
) -> tuple[int, np.ndarray]:
    """Convert a list of LAS tiles into blocks.

    Returns the total block count and a (NUM_CLASSES,) array of class
    point-counts aggregated across all saved blocks.
    """
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    n_blocks = 0

    for i, path in enumerate(las_paths):
        t0 = time.time()
        coord, strength, segment = read_tile(path)
        blocks = split_into_blocks(coord, strength, segment, block_size, min_points)

        tile_name = path.stem
        for block in blocks:
            block_name = f"{tile_name}_{block['grid_idx']}"
            save_block(block, output_dir / block_name)

            codes, counts = np.unique(block["segment"], return_counts=True)
            for c, cnt in zip(codes, counts):
                class_counts[int(c)] += int(cnt)
            n_blocks += 1

        elapsed = time.time() - t0
        logger.info(
            "  [%d/%d] %s: %s pts -> %d blocks (%.1fs)",
            i + 1,
            len(las_paths),
            tile_name,
            f"{len(coord):,}",
            len(blocks),
            elapsed,
        )

    return n_blocks, class_counts


def compute_class_weights(class_counts: np.ndarray) -> list[float]:
    """Inverse-frequency class weights (ignored class gets weight 0)."""
    total = class_counts.sum()
    weights = np.zeros(NUM_CLASSES, dtype=np.float64)

    for c in range(NUM_CLASSES):
        if c == IGNORE_INDEX or class_counts[c] == 0:
            continue
        weights[c] = 1.0 / (class_counts[c] / total + 1e-6)

    # Normalise so non-zero weights sum to (NUM_CLASSES - 1)
    active = weights > 0
    if active.any():
        weights[active] *= (NUM_CLASSES - 1) / weights[active].sum()

    return weights.tolist()


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess DALES LAS tiles into Pointcept .npy blocks",
    )
    parser.add_argument(
        "--input_dir",
        default="dales_las",
        help="DALES root with train/ and test/ subdirs (default: dales_las)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/dales_ptv3",
        help="Output directory (default: data/dales_ptv3)",
    )
    parser.add_argument(
        "--block_size",
        type=float,
        default=50.0,
        help="Spatial block size in metres (default: 50)",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=100,
        help="Discard blocks with fewer points (default: 100)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of train tiles held out for validation (default: 0.15)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for val split")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    train_dir = input_dir / "train"
    test_dir = input_dir / "test"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_files = sorted(train_dir.glob("*.las"))
    test_files = sorted(test_dir.glob("*.las"))
    logger.info(
        "Found %d train tiles, %d test tiles", len(train_files), len(test_files)
    )

    # ── train / val split ─────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    n_val = max(1, int(len(train_files) * args.val_ratio))
    indices = rng.permutation(len(train_files))
    val_indices = set(indices[:n_val].tolist())

    val_files = [f for i, f in enumerate(train_files) if i in val_indices]
    actual_train = [f for i, f in enumerate(train_files) if i not in val_indices]
    logger.info(
        "Split: %d train, %d val, %d test",
        len(actual_train),
        len(val_files),
        len(test_files),
    )
    logger.info("Val tiles: %s", [f.stem for f in val_files])

    # ── process each split ────────────────────────────────────────────
    train_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for split_name, split_files in [
        ("train", actual_train),
        ("val", val_files),
        ("test", test_files),
    ]:
        logger.info("\nProcessing %s (%d tiles) ...", split_name, len(split_files))
        n_blocks, counts = process_tiles(
            split_files,
            output_dir / split_name,
            args.block_size,
            args.min_points,
        )
        logger.info("  -> %d blocks", n_blocks)
        if split_name == "train":
            train_counts = counts

    # ── class weights & metadata ──────────────────────────────────────
    class_weights = compute_class_weights(train_counts)

    metadata = {
        "dataset": "DALES",
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "class_names": {str(k): v for k, v in DALES_CLASSES.items()},
        "class_weights": class_weights,
        "class_counts": train_counts.tolist(),
        "block_size": args.block_size,
        "train_tiles": [f.stem for f in actual_train],
        "val_tiles": [f.stem for f in val_files],
        "test_tiles": [f.stem for f in test_files],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\nMetadata saved to %s", meta_path)
    logger.info("Class distribution (train):")
    total = train_counts.sum()
    for c in range(NUM_CLASSES):
        name = DALES_CLASSES[c]
        pct = train_counts[c] / total * 100 if total > 0 else 0
        logger.info(
            "  %d %-15s %12s (%5.1f%%)  weight=%.4f",
            c,
            name,
            f"{train_counts[c]:,}",
            pct,
            class_weights[c],
        )

    logger.info("\nDone!  Next step:")
    logger.info("  python train_ptv3_dales.py --data_root %s", output_dir)


if __name__ == "__main__":
    main()
