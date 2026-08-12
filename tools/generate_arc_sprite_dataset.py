# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# ruff: noqa: DOC201, DOC501

"""Generate the effective-rendered ARC sprite-child dataset.

Run the full dataset generator from the repository root::

    uv run --no-sync python tools/generate_arc_sprite_dataset.py --overwrite

Generate one game in a temporary directory::

    uv run --no-sync python tools/generate_arc_sprite_dataset.py \
        --games ar25 --output-dir /tmp/ar25-sprites --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DISPLAY_SIZE = 64
COMPACT_MAX_DIM = 8
COMPACT_MIN_BBOX_FRACTION = 0.5
DATASET_VERSION = 1
DEFAULT_DATASET_SUBPATH = Path("arc/sprite_children/v1")

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
    """Return the values used by the compact-object filter."""
    array = np.asarray(pixels)
    visible = array >= 0
    rows, columns = np.nonzero(visible)
    bbox_area = (
        int(rows.max() - rows.min() + 1) * int(columns.max() - columns.min() + 1)
        if len(rows)
        else 0
    )
    return {
        "height": array.shape[0],
        "width": array.shape[1],
        "bbox_fraction": visible.sum() / bbox_area if bbox_area else 0.0,
        "component_count": count_components(visible),
        "raw_signature": pixel_signature(array),
    }


def is_compact_candidate(pixels: Any) -> bool:
    """Return whether raw pixels describe a compact child object."""
    metrics = compact_metrics(pixels)
    return (
        max(metrics["height"], metrics["width"]) <= COMPACT_MAX_DIM
        and metrics["component_count"] == 1
        and metrics["bbox_fraction"] >= COMPACT_MIN_BBOX_FRACTION
    )


def camera_cell_scale(grid_size: Any, display_size: int = DISPLAY_SIZE) -> int:
    """Return the camera's integer pixels-per-cell scale."""
    width, height = grid_size
    return min(display_size // width, display_size // height)


def canonicalize_pixels(pixels: Any) -> np.ndarray:
    """Represent every transparent value as ``-1``."""
    array = np.asarray(pixels)
    return np.where(array < 0, -1, array).astype(np.int8, copy=False)


def effective_pixels(rendered: Any, grid_size: Any) -> np.ndarray:
    """Expand logical cells to their displayed size."""
    pixels = canonicalize_pixels(rendered)
    scale = camera_cell_scale(grid_size)
    return np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)


def effective_sprite_pixels(sprite: Any, grid_size: Any) -> np.ndarray:
    """Render and expand a Sprite."""
    return effective_pixels(sprite.render(), grid_size)


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


def write_png(path: Path, pixels: Any) -> None:
    """Write an RGBA preview."""
    from PIL import Image  # noqa: PLC0415

    Image.fromarray(rgba_preview(pixels), mode="RGBA").save(path)


def discover_games(games_root: Path, requested: list[str] | None) -> list[Path]:
    """Find game modules, optionally restricted by game or map ID."""
    paths = sorted(games_root.glob("*/*/*.py"))
    if not requested:
        return paths
    requested = set(requested)
    return [
        path
        for path in paths
        if path.parent.parent.name in requested
        or f"{path.parent.parent.name}-{path.parent.name}" in requested
    ]


def load_game_module(path: Path, ordinal: int) -> Any:
    """Load one local game module."""
    name = f"_arc_sprite_dataset_{ordinal}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def map_identity(path: Path) -> tuple[str, str, str]:
    """Return game, version, and map IDs."""
    game_id = path.parent.parent.name
    version = path.parent.name
    return game_id, version, f"{game_id}-{version}"


def definition_record(
    path: Path, map_id: str, game_id: str, version: str, key: str, sprite: Any
) -> dict[str, Any]:
    """Describe one dictionary-defined Sprite."""
    pixels = canonicalize_pixels(sprite.pixels)
    metrics = compact_metrics(pixels)
    return {
        "definition_id": f"{map_id}:{key}",
        "source_path": str(path),
        "map_id": map_id,
        "game_id": game_id,
        "version": version,
        "base_label": str(sprite.name),
        "raw_signature": metrics["raw_signature"],
        "compact": (
            max(metrics["height"], metrics["width"]) <= COMPACT_MAX_DIM
            and metrics["component_count"] == 1
            and metrics["bbox_fraction"] >= COMPACT_MIN_BBOX_FRACTION
        ),
    }


def occurrence_record(
    definition: dict[str, Any],
    level: Any,
    level_index: int,
    instance_index: int,
    sprite: Any,
) -> dict[str, Any]:
    """Describe one placed Sprite."""
    grid_size = [int(value) for value in level.grid_size]
    return {
        "definition_id": definition["definition_id"],
        "base_label": definition["base_label"],
        "raw_signature": definition["raw_signature"],
        "map_id": definition["map_id"],
        "game_id": definition["game_id"],
        "version": definition["version"],
        "level_index": level_index,
        "level_name": str(getattr(level, "name", "Level")),
        "instance_index": instance_index,
        "grid_size": grid_size,
        "cell_scale_px": camera_cell_scale(grid_size),
        "x": int(sprite.x),
        "y": int(sprite.y),
        "scale": int(sprite.scale),
        "rotation": int(sprite.rotation),
        "mirror_ud": bool(sprite.mirror_ud),
        "mirror_lr": bool(sprite.mirror_lr),
        "sprite": sprite,
    }


def collect_source(
    game_paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect Sprite definitions, placements, and map summaries."""
    definitions = []
    occurrences = []
    map_summaries = []
    for ordinal, path in enumerate(game_paths):
        module = load_game_module(path, ordinal)
        game_id, version, map_id = map_identity(path)
        map_definitions = [
            definition_record(path, map_id, game_id, version, str(key), sprite)
            for key, sprite in module.sprites.items()
        ]
        by_name = {row["base_label"]: row for row in map_definitions}
        map_occurrences = [
            occurrence_record(
                by_name[str(sprite.name)], level, level_index, index, sprite
            )
            for level_index, level in enumerate(module.levels, start=1)
            for index, sprite in enumerate(level.get_sprites())
        ]
        definitions.extend(map_definitions)
        occurrences.extend(map_occurrences)
        map_summaries.append(
            {
                "map_id": map_id,
                "definitions": len(map_definitions),
                "levels": len(module.levels),
                "instances": len(map_occurrences),
                "compact_definitions": sum(row["compact"] for row in map_definitions),
            }
        )
    return definitions, occurrences, map_summaries


def group_by(
    rows: Iterable[dict[str, Any]], field: str
) -> dict[Any, list[dict[str, Any]]]:
    """Group dictionaries by one field."""
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return groups


def raw_ambiguous_groups(
    compact_definitions: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Find raw signatures shared by multiple base labels."""
    return {
        signature: rows
        for signature, rows in group_by(compact_definitions, "raw_signature").items()
        if len({row["base_label"] for row in rows}) > 1
    }


def excluded_definition_rows(
    definitions: list[dict[str, Any]], occurrences: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Exclude ambiguous and unused compact definitions."""
    compact = [row for row in definitions if row["compact"]]
    ambiguous = raw_ambiguous_groups(compact)
    counts = Counter(row["definition_id"] for row in occurrences)
    excluded = []
    selected = set()
    for definition in sorted(compact, key=lambda row: row["definition_id"]):
        signature = definition["raw_signature"]
        count = counts[definition["definition_id"]]
        if signature in ambiguous:
            excluded.append(
                {
                    "kind": "definition",
                    "reason": "raw_ambiguous_signature",
                    "definition_id": definition["definition_id"],
                    "base_label": definition["base_label"],
                    "raw_signature": signature,
                    "conflicting_base_labels": sorted(
                        {row["base_label"] for row in ambiguous[signature]}
                    ),
                    "source_occurrence_count": count,
                }
            )
        elif not count:
            excluded.append(
                {
                    "kind": "definition",
                    "reason": "unused_definition",
                    "definition_id": definition["definition_id"],
                    "base_label": definition["base_label"],
                    "raw_signature": signature,
                    "source_occurrence_count": 0,
                }
            )
        else:
            selected.add(definition["definition_id"])
    return excluded, selected


SOURCE_FIELDS = (
    "definition_id",
    "raw_signature",
    "map_id",
    "game_id",
    "version",
    "level_index",
    "level_name",
    "instance_index",
    "grid_size",
    "cell_scale_px",
    "x",
    "y",
    "scale",
    "rotation",
    "mirror_ud",
    "mirror_lr",
)


def source_provenance(occurrence: dict[str, Any]) -> dict[str, Any]:
    """Remove the live Sprite from an occurrence record."""
    return {key: occurrence[key] for key in SOURCE_FIELDS}


def build_variants(
    occurrences: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render, deduplicate, and exclude cross-label collisions."""
    variants = {}
    for occurrence in occurrences:
        pixels = effective_sprite_pixels(occurrence["sprite"], occurrence["grid_size"])
        signature = pixel_signature(pixels)
        variant = variants.setdefault(
            (occurrence["base_label"], signature),
            {
                "base_label": occurrence["base_label"],
                "effective_signature": signature,
                "pixels": pixels,
                "occurrences": [],
            },
        )
        variant["occurrences"].append(source_provenance(occurrence))

    retained = []
    excluded = []
    for signature, group in sorted(
        group_by(variants.values(), "effective_signature").items()
    ):
        labels = sorted({row["base_label"] for row in group})
        if len(labels) == 1:
            retained.extend(group)
            continue
        for variant in sorted(group, key=lambda row: row["base_label"]):
            excluded.append(
                {
                    "kind": "effective_variant",
                    "reason": "effective_ambiguous_collision",
                    "base_label": variant["base_label"],
                    "effective_signature": signature,
                    "effective_shape": list(variant["pixels"].shape),
                    "conflicting_base_labels": labels,
                    "source_occurrence_count": len(variant["occurrences"]),
                    "sources": variant["occurrences"],
                }
            )
    return sorted(
        retained, key=lambda row: (row["base_label"], row["effective_signature"])
    ), excluded


def variant_manifest_row(variant: dict[str, Any]) -> dict[str, Any]:
    """Build one manifest row."""
    pixels = variant["pixels"]
    label = object_label(variant["base_label"], pixels)
    sources = sorted(
        variant["occurrences"],
        key=lambda row: (row["map_id"], row["level_index"], row["instance_index"]),
    )
    return {
        "sample_id": label,
        "object_label": label,
        "base_label": variant["base_label"],
        "raw_signatures": sorted({row["raw_signature"] for row in sources}),
        "effective_signature": variant["effective_signature"],
        "effective_shape": list(pixels.shape),
        "dtype": str(pixels.dtype),
        "npy_path": f"sprites/{label}.npy",
        "png_path": f"previews/{label}.png",
        "cell_scales_px": sorted({row["cell_scale_px"] for row in sources}),
        "grid_sizes": [
            list(size) for size in sorted({tuple(row["grid_size"]) for row in sources})
        ],
        "source_occurrence_count": len(sources),
        "sources": sources,
    }


def prepare_output_dir(output_dir: Path, overwrite: bool = False) -> None:
    """Create an empty dataset directory."""
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    (output_dir / "sprites").mkdir(parents=True)
    (output_dir / "previews").mkdir()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write deterministic JSON Lines."""
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_report(
    path: Path,
    dataset: dict[str, Any],
    map_summaries: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> None:
    """Write a compact human-readable report."""
    exclusions = Counter(row["reason"] for row in excluded)
    lines = [
        "# ARC sprite child dataset",
        "",
        "Variable-shaped effective-rendered Sprite canvases.",
        "",
        "## Counts",
        "",
        *(
            f"- {key.replace('_', ' ')}: {value}"
            for key, value in dataset["counts"].items()
        ),
        "",
        "## Exclusions",
        "",
        *(f"- {reason}: {count}" for reason, count in sorted(exclusions.items())),
        "",
        "## Maps",
        "",
        "| map | definitions | levels | instances | compact |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            f"| {row['map_id']} | {row['definitions']} | {row['levels']} | "
            f"{row['instances']} | {row['compact_definitions']} |"
            for row in map_summaries
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_dataset(
    games_root: Path,
    output_dir: Path,
    requested_games: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate the sprite-child dataset."""
    game_paths = discover_games(games_root, requested_games)
    definitions, occurrences, map_summaries = collect_source(game_paths)
    excluded_definitions, selected_ids = excluded_definition_rows(
        definitions, occurrences
    )
    variants, excluded_effective = build_variants(
        row for row in occurrences if row["definition_id"] in selected_ids
    )
    excluded = excluded_definitions + excluded_effective
    manifest = [variant_manifest_row(variant) for variant in variants]

    prepare_output_dir(output_dir, overwrite)
    for variant, row in zip(variants, manifest):
        np.save(output_dir / row["npy_path"], variant["pixels"], allow_pickle=False)
        write_png(output_dir / row["png_path"], variant["pixels"])

    compact = [row for row in definitions if row["compact"]]
    ambiguous = raw_ambiguous_groups(compact)
    collision_count = len(excluded_effective)
    dataset = {
        "dataset": "arc_sprite_children",
        "version": DATASET_VERSION,
        "source": {
            "games_root": str(games_root.resolve()),
            "maps": [row["map_id"] for row in map_summaries],
            "game_paths": [str(path.resolve()) for path in game_paths],
        },
        "filter": {
            "compact_max_dim": COMPACT_MAX_DIM,
            "compact_min_bbox_fraction": COMPACT_MIN_BBOX_FRACTION,
            "compact_component_connectivity": 4,
            "raw_ambiguity_key": "raw_signature + multiple base_label values",
            "unused_definitions_excluded": True,
            "effective_cross_label_collisions_excluded": True,
        },
        "rendering": {
            "display_size": DISPLAY_SIZE,
            "sprite_render": "Sprite.render()",
            "camera_scale": "min(64 // grid_width, 64 // grid_height)",
            "cell_scaling": "nearest_neighbor_repeat",
            "canvas": "full_rendered_canvas_preserved",
            "negative_value": -1,
            "visible_palette_values": list(range(16)),
        },
        "palette_rgba": ARC_RGBA_PALETTE.tolist(),
        "counts": {
            "maps": len(map_summaries),
            "definitions": len(definitions),
            "compact_definitions": len(compact),
            "raw_unique_signatures": len({row["raw_signature"] for row in compact}),
            "raw_ambiguous_signature_groups": len(ambiguous),
            "raw_ambiguous_definitions": sum(map(len, ambiguous.values())),
            "unused_definitions": sum(
                row["reason"] == "unused_definition" for row in excluded_definitions
            ),
            "used_definitions": len(selected_ids),
            "base_labels": len({row["base_label"] for row in variants}),
            "effective_variants_before_cross_label_exclusion": len(variants)
            + collision_count,
            "effective_cross_label_collision_variants": collision_count,
            "object_labels": len(manifest),
            "source_occurrences_before_effective_exclusion": sum(
                len(row["occurrences"]) for row in variants
            )
            + sum(row["source_occurrence_count"] for row in excluded_effective),
            "source_occurrences": sum(
                row["source_occurrence_count"] for row in manifest
            ),
        },
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(output_dir / "manifest.jsonl", manifest)
    write_jsonl(
        output_dir / "excluded.jsonl",
        sorted(
            excluded,
            key=lambda row: (
                row["reason"],
                row.get("definition_id", ""),
                row.get("base_label", ""),
                row.get("effective_signature", ""),
            ),
        ),
    )
    write_report(output_dir / "report.md", dataset, map_summaries, excluded)
    return dataset


def default_output_dir() -> Path:
    """Return the default location beneath ``MONTY_DATA``."""
    return Path(os.environ["MONTY_DATA"]).expanduser() / DEFAULT_DATASET_SUBPATH


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
    args.output_dir = args.output_dir or default_output_dir()
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
        f"{counts['source_occurrences']} occurrences at {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
