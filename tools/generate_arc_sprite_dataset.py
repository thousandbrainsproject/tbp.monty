# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# ruff: noqa: DOC201, DOC501

"""Generate clean ARC sprite children and their initial-frame oracle regions.

Run from the repository root::

    uv run --no-sync python tools/generate_arc_sprite_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DISPLAY_SIZE = 64
COMPACT_MAX_DIM = 8
COMPACT_MIN_BBOX_FRACTION = 0.5
DATASET_VERSION = 2
DEFAULT_DATASET_SUBPATH = Path("arc/sprite_children/v2")

ARC_RGBA_PALETTE = np.array(
    [
        (255, 255, 255, 255),
        (204, 204, 204, 255),
        (153, 153, 153, 255),
        (102, 102, 102, 255),
        (51, 51, 51, 255),
        (0, 0, 0, 255),
        (229, 58, 163, 255),
        (255, 123, 204, 255),
        (249, 60, 49, 255),
        (30, 147, 255, 255),
        (136, 216, 241, 255),
        (255, 220, 0, 255),
        (255, 133, 27, 255),
        (146, 18, 49, 255),
        (79, 204, 48, 255),
        (163, 86, 214, 255),
    ],
    dtype=np.uint8,
)


def pixel_signature(pixels: Any) -> str:
    """Hash a palette array independently of its source dtype."""
    array = np.ascontiguousarray(pixels, dtype=np.int16)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b":")
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonicalize_pixels(pixels: Any) -> np.ndarray:
    """Represent every transparent value as ``-1``."""
    array = np.asarray(pixels)
    return np.where(array < 0, -1, array).astype(np.int8, copy=False)


def count_components(mask: np.ndarray) -> int:
    """Count four-connected components."""
    remaining = set(zip(*np.nonzero(mask)))
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            row, column = stack.pop()
            neighbors = {
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            }
            found = remaining & neighbors
            remaining -= found
            stack.extend(found)
    return components


def compact_metrics(pixels: Any) -> dict[str, Any]:
    """Return the values used by the compact-child filter."""
    array = canonicalize_pixels(pixels)
    visible = array >= 0
    rows, columns = np.nonzero(visible)
    bbox_area = (
        int(rows.max() - rows.min() + 1) * int(columns.max() - columns.min() + 1)
        if len(rows)
        else 0
    )
    height, width = array.shape
    fraction = visible.sum() / bbox_area if bbox_area else 0.0
    components = count_components(visible)
    return {
        "raw_signature": pixel_signature(array),
        "compact": (
            max(height, width) <= COMPACT_MAX_DIM
            and components == 1
            and fraction >= COMPACT_MIN_BBOX_FRACTION
        ),
        "component_count": components,
    }


def camera_cell_scale(grid_size: Any, display_size: int = DISPLAY_SIZE) -> int:
    """Return the camera's integer pixels-per-cell scale."""
    width, height = grid_size
    return min(display_size // width, display_size // height)


def effective_pixels(rendered: Any, grid_size: Any) -> np.ndarray:
    """Expand logical cells to their displayed size."""
    pixels = canonicalize_pixels(rendered)
    scale = camera_cell_scale(grid_size)
    return np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)


def object_label(base_label: str, pixels: Any) -> str:
    """Build a scale- and pattern-specific label."""
    height, width = np.asarray(pixels).shape
    return f"{base_label}__{height}x{width}__{pixel_signature(pixels)[:12]}"


def rgba_preview(pixels: Any) -> np.ndarray:
    """Convert palette indices to a transparent RGBA preview."""
    pixels = canonicalize_pixels(pixels)
    rgba = np.zeros((*pixels.shape, 4), dtype=np.uint8)
    visible = pixels >= 0
    rgba[visible] = ARC_RGBA_PALETTE[pixels[visible]]
    return rgba


def discover_games(games_root: Path, requested: list[str] | None) -> list[Path]:
    """Find game modules, optionally restricted by game or map ID."""
    paths = sorted(games_root.glob("*/*/*.py"))
    if not requested:
        return paths
    requested_set = set(requested)
    return [
        path
        for path in paths
        if path.parent.parent.name in requested_set
        or f"{path.parent.parent.name}-{path.parent.name}" in requested_set
    ]


def load_game_module(path: Path, ordinal: int) -> Any:
    """Load one local game module."""
    name = f"_arc_sprite_dataset_{ordinal}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ARC game module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def display_sprite(sprite: Any, camera: Any) -> dict[str, Any]:
    """Project one sprite into the 64x64 display before layer occlusion."""
    pixels = canonicalize_pixels(sprite.render())
    scale = min(DISPLAY_SIZE // camera.width, DISPLAY_SIZE // camera.height)
    x_offset = (DISPLAY_SIZE - camera.width * scale) // 2
    y_offset = (DISPLAY_SIZE - camera.height * scale) // 2
    rel_x, rel_y = sprite.x - camera.x, sprite.y - camera.y

    x0, x1 = max(0, rel_x), min(camera.width, rel_x + pixels.shape[1])
    y0, y1 = max(0, rel_y), min(camera.height, rel_y + pixels.shape[0])
    mask = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE), dtype=bool)
    colors = np.full((DISPLAY_SIZE, DISPLAY_SIZE), -1, dtype=np.int8)
    bbox = [0, 0, 0, 0]
    if x0 < x1 and y0 < y1:
        source = pixels[y0 - rel_y : y1 - rel_y, x0 - rel_x : x1 - rel_x]
        source = np.repeat(np.repeat(source, scale, axis=0), scale, axis=1)
        dx0, dy0 = x_offset + x0 * scale, y_offset + y0 * scale
        dx1, dy1 = dx0 + source.shape[1], dy0 + source.shape[0]
        colors[dy0:dy1, dx0:dx1] = source
        mask[dy0:dy1, dx0:dx1] = source >= 0
        bbox = [dx0, dy0, dx1, dy1]

    return {
        "pixels": effective_pixels(pixels, (camera.width, camera.height)),
        "mask": mask,
        "colors": colors,
        "bbox": bbox,
        "source_pixel_count": int((pixels >= 0).sum()) * scale * scale,
    }


def initial_frame_regions(
    sprites: list[Any], camera: Any, source_ids: list[str]
) -> list[dict[str, Any]]:
    """Measure source-pixel visibility after clipping, layering, and interfaces."""
    projected = [display_sprite(sprite, camera) for sprite in sprites]
    owner = np.full((DISPLAY_SIZE, DISPLAY_SIZE), -1, dtype=int)
    for index in sorted(range(len(sprites)), key=lambda item: sprites[item].layer):
        if sprites[index].is_visible:
            owner[projected[index]["mask"]] = index

    final_frame = canonicalize_pixels(camera.render(sprites))
    regions = []
    for index, (sprite, source_id, display) in enumerate(
        zip(sprites, source_ids, projected)
    ):
        source_count = display["source_pixel_count"]
        visible = (
            display["mask"] & (owner == index) & (display["colors"] == final_frame)
        )
        visible_count = int(visible.sum()) if sprite.is_visible else 0
        x0, y0, x1, y1 = display["bbox"]
        visible_mask = visible[y0:y1, x0:x1]
        fraction = visible_count / source_count if source_count else 0.0
        regions.append(
            {
                "source_sprite_id": source_id,
                "bbox": display["bbox"],
                "layer": int(sprite.layer),
                "visible_mask": visible_mask.tolist(),
                "visible_pixel_count": visible_count,
                "visible_fraction": fraction,
                "fully_visible": bool(source_count and visible_count == source_count),
                "pixels": display["pixels"],
            }
        )
    return regions


def collect_clean_occurrences(
    game_paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Collect fully visible compact occurrences from frozen initial frames."""
    from arcengine import ARCBaseGame  # noqa: PLC0415

    clean = []
    counts: Counter[str] = Counter()
    for ordinal, path in enumerate(game_paths):
        module = load_game_module(path, ordinal)
        game_class = next(
            value
            for value in vars(module).values()
            if inspect.isclass(value)
            and value.__module__ == module.__name__
            and issubclass(value, ARCBaseGame)
        )
        game_id, version = path.parent.parent.name, path.parent.name
        map_id = f"{game_id}-{version}"
        invisible_definitions = {
            (str(sprite.name), pixel_signature(canonicalize_pixels(sprite.pixels)))
            for sprite in module.sprites.values()
            if not sprite.is_visible
        }

        for level_index in range(len(module.levels)):
            game = game_class()
            if level_index:
                game.set_level(level_index)
            sprites = game.current_level.get_sprites()
            source_ids = [
                f"{map_id}:{level_index + 1}:{index}:{sprite.name}"
                for index, sprite in enumerate(sprites)
            ]
            regions = initial_frame_regions(sprites, game.camera, source_ids)
            counts["occurrences"] += len(sprites)
            for sprite, region in zip(sprites, regions):
                metrics = compact_metrics(sprite.pixels)
                definition_key = (str(sprite.name), metrics["raw_signature"])
                if definition_key in invisible_definitions:
                    counts["invisible_definition_occurrences"] += 1
                elif not sprite.is_visible:
                    counts["invisible_occurrences"] += 1
                elif not metrics["compact"]:
                    counts["noncompact_occurrences"] += 1
                elif not region["fully_visible"]:
                    counts["occluded_occurrences"] += 1
                else:
                    clean.append(
                        {
                            "base_label": str(sprite.name),
                            "raw_signature": metrics["raw_signature"],
                            "pixels": region.pop("pixels"),
                            "region": region,
                        }
                    )
                    counts["clean_occurrences"] += 1
    return clean, counts


def build_variants(
    occurrences: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate clean occurrences without discarding their labels."""
    variants = {}
    for occurrence in occurrences:
        signature = pixel_signature(occurrence["pixels"])
        variant = variants.setdefault(
            (occurrence["base_label"], signature),
            {
                "base_label": occurrence["base_label"],
                "effective_signature": signature,
                "pixels": occurrence["pixels"],
                "regions": [],
            },
        )
        variant["regions"].append(occurrence["region"])
    return sorted(
        variants.values(),
        key=lambda row: (row["base_label"], row["effective_signature"]),
    )


def manifest_row(variant: dict[str, Any]) -> dict[str, Any]:
    """Return the minimal artifact and oracle-region contract."""
    label = object_label(variant["base_label"], variant["pixels"])
    regions = sorted(variant["regions"], key=lambda row: row["source_sprite_id"])
    return {
        "object_label": label,
        "npy_path": f"sprites/{label}.npy",
        "png_path": f"previews/{label}.png",
        "oracle_regions": regions,
    }


def prepare_output_dir(output_dir: Path, overwrite: bool = False) -> None:
    """Create an empty dataset directory."""
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    (output_dir / "sprites").mkdir(parents=True)
    (output_dir / "previews").mkdir()


def generate_dataset(
    games_root: Path,
    output_dir: Path,
    requested_games: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate the clean sprite-child dataset."""
    game_paths = discover_games(games_root, requested_games)
    occurrences, counts = collect_clean_occurrences(game_paths)
    variants = build_variants(occurrences)
    manifest = [manifest_row(variant) for variant in variants]

    prepare_output_dir(output_dir, overwrite)
    from PIL import Image  # noqa: PLC0415

    for variant, row in zip(variants, manifest):
        np.save(output_dir / row["npy_path"], variant["pixels"], allow_pickle=False)
        Image.fromarray(rgba_preview(variant["pixels"]), mode="RGBA").save(
            output_dir / row["png_path"]
        )

    counts["object_labels"] = len(manifest)
    dataset = {
        "dataset": "arc_sprite_children",
        "version": DATASET_VERSION,
        "maps": [
            f"{path.parent.parent.name}-{path.parent.name}" for path in game_paths
        ],
        "filters": {
            "compact_max_dim": COMPACT_MAX_DIM,
            "compact_min_bbox_fraction": COMPACT_MIN_BBOX_FRACTION,
            "definition_visible": True,
            "fully_visible_initial_frame": True,
        },
        "counts": dict(counts),
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in manifest
        ),
        encoding="utf-8",
    )
    report = [
        "# ARC sprite child dataset",
        "",
        "Only fully visible compact occurrences from frozen initial frames are used.",
        "",
        "## Counts",
        "",
        *(f"- {key.replace('_', ' ')}: {value}" for key, value in counts.items()),
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "arc_agi/games",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--games", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir or (
        Path(os.environ["MONTY_DATA"]).expanduser() / DEFAULT_DATASET_SUBPATH
    )
    return args


def main(argv: list[str] | None = None) -> int:
    """Generate the dataset from the command line."""
    args = parse_args(argv)
    dataset = generate_dataset(
        args.games_root.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.games,
        args.overwrite,
    )
    counts = dataset["counts"]
    print(
        f"Generated {counts['object_labels']} ARC sprite objects from "
        f"{counts['clean_occurrences']} clean occurrences at {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
