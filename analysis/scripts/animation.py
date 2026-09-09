# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure



def save_animation(
    fig: Figure,
    update_frame: Callable[[int], list],
    n_frames: int,
    out_path: Path,
    fps: int,
    fmt: str = "gif",
) -> Path:
    """Animate a figure and save it, creating the output directory.

    Uses ``blit=False``: saving redraws the whole figure anyway, and blitting
    only complicates which artists appear in the saved frames.

    Args:
        fig: The assembled figure.
        update_frame: Called with the step number to redraw the figure.
        n_frames: How many steps to animate.
        out_path: Where to save the animation.
        fps: Frames per second of the saved animation.
        fmt: "gif" (PillowWriter) or "mp4" (FFMpegWriter, needs ffmpeg on
            the PATH).

    Returns:
        The saved path.

    Raises:
        ValueError: If fmt is not "gif" or "mp4".
    """
    if fmt == "gif":
        writer = PillowWriter(fps=fps)
    elif fmt == "mp4":
        writer = FFMpegWriter(fps=fps)
    else:
        raise ValueError(f"fmt must be 'gif' or 'mp4', got {fmt!r}")
    anim = FuncAnimation(fig, update_frame, frames=n_frames, blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=writer)

    print(f"Animation saved to: {out_path}")
    plt.close(fig)

    return out_path
