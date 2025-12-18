"""Script to generate diagonal line pattern images.

Generates images with parallel diagonal lines on white background at
three resolutions: 64x64, 32x32, and 16x16.
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_diagonal_pattern(
    size: int,
    line_color: tuple = (0, 0, 0),
    bg_color: tuple = (255, 255, 255),
    line_thickness: int = 1,
    line_length: float = None,
    line_spacing: float = None,
    num_lines: int = None,
    # Global = where segment centers lie (top-left -> bottom-right).
    global_angle_deg: float = 45.0,
    # Segment = orientation of each short stroke (bottom-left -> top-right).
    segment_angle_deg: float = -45.0,
    margin_ratio: float = 0.15,
    supersample: int = 8,
) -> np.ndarray:
    """Generate an image with short diagonal strokes aligned along a global diagonal.

    The goal pattern is:
      - A clear *global* diagonal running top-left -> bottom-right (via stroke centers).
      - *Local* short strokes running bottom-left -> top-right.

    Supersampling is used to avoid checkerboard/dotted aliasing at low resolutions.

    Args:
        size: Output image size (size x size)
        line_color: RGB tuple for line color
        bg_color: RGB tuple for background color
        line_thickness: Stroke thickness in pixels (at output resolution)
        line_length: Length of each stroke (if None, auto-calculated)
        line_spacing: Spacing between stroke centers along the global diagonal (if None, auto-calculated)
        num_lines: Number of strokes to draw (if None, auto-calculated)
        global_angle_deg: Direction of the global diagonal (default: 45 = TL->BR in image coords)
        segment_angle_deg: Direction of each short stroke (default: -45 = BL->TR in image coords)
        margin_ratio: Ratio of margin from edges
        supersample: Render scale factor to reduce aliasing (>=1)

    Returns:
        RGB image array of shape (size, size, 3) with dtype uint8
    """
    supersample = max(1, int(supersample))
    S = int(size * supersample)

    # Create background at supersampled resolution
    image_hi = np.full((S, S, 3), bg_color, dtype=np.uint8)

    # Scale drawing parameters to supersampled resolution
    thickness_hi = max(1, int(round(line_thickness * supersample)))
    margin_hi = int(round(size * margin_ratio * supersample))

    if line_length is None:
        # Keep strokes compact so adjacent strokes don't visually merge after downsampling.
        line_length = max(5 * line_thickness, size * 0.10)
    if line_spacing is None:
        # Increase spacing to avoid checkerboard / moiré when downsampling.
        line_spacing = max(3.0 * line_thickness, size * 0.060)

    line_length_hi = float(line_length) * supersample
    line_spacing_hi = float(line_spacing) * supersample

    # Convert angles to radians
    g_rad = np.deg2rad(global_angle_deg)
    s_rad = np.deg2rad(segment_angle_deg)

    # Direction vectors
    dir_global = np.array([np.cos(g_rad), np.sin(g_rad)], dtype=np.float32)
    dir_global /= np.linalg.norm(dir_global) + 1e-9

    dir_seg = np.array([np.cos(s_rad), np.sin(s_rad)], dtype=np.float32)
    dir_seg /= np.linalg.norm(dir_seg) + 1e-9

    # Define a start/end segment for the global diagonal within margins.
    # Use intersection with a TL->BR style diagonal when global_angle is 45,
    # but keep it general by projecting across the image.
    center_hi = np.array([S / 2.0, S / 2.0], dtype=np.float32)

    # Compute how far we can go in +/- global direction before hitting margin box.
    # We find t such that center + t*dir is within [margin, S-1-margin] for both axes.
    lo = margin_hi
    hi = (S - 1) - margin_hi

    t_candidates = []
    for axis in (0, 1):
        d = dir_global[axis]
        if abs(d) < 1e-9:
            continue
        t1 = (lo - center_hi[axis]) / d
        t2 = (hi - center_hi[axis]) / d
        t_candidates.extend([t1, t2])

    # Evaluate candidates and take min/max valid t where point remains inside on both axes
    ts_valid = []
    for t in t_candidates:
        p = center_hi + t * dir_global
        if lo - 1e-6 <= p[0] <= hi + 1e-6 and lo - 1e-6 <= p[1] <= hi + 1e-6:
            ts_valid.append(t)

    if len(ts_valid) < 2:
        # Fallback: just use corners
        start_c = np.array([lo, lo], dtype=np.float32)
        end_c = np.array([hi, hi], dtype=np.float32)
    else:
        t_min = float(min(ts_valid))
        t_max = float(max(ts_valid))
        start_c = center_hi + t_min * dir_global
        end_c = center_hi + t_max * dir_global

    global_len = float(np.linalg.norm(end_c - start_c))
    if num_lines is None:
        num_lines = int(global_len / line_spacing_hi) + 1

    # OpenCV draw expects integer tuples (x, y)
    image_bgr = cv2.cvtColor(image_hi, cv2.COLOR_RGB2BGR)
    bgr_color = (line_color[2], line_color[1], line_color[0])

    half_len = 0.5 * line_length_hi

    for i in range(num_lines):
        c = start_c + (i * line_spacing_hi) * dir_global

        start = c - half_len * dir_seg
        end = c + half_len * dir_seg

        start_int = (int(round(start[0])), int(round(start[1])))
        end_int = (int(round(end[0])), int(round(end[1])))

        cv2.line(
            image_bgr,
            start_int,
            end_int,
            bgr_color,
            thickness_hi,
            lineType=cv2.LINE_AA,
        )

    image_hi = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Downsample back to requested size
    if supersample > 1:
        image = cv2.resize(image_hi, (size, size), interpolation=cv2.INTER_AREA)
    else:
        image = image_hi

    return image


def main():
    """Generate diagonal line pattern images at three resolutions."""
    output_dir = Path("results/diagonal_line_patterns")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base parameters for 64x64 (reference resolution)
    base_size = 64
    base_thickness = 1
    base_line_length = None  # Auto-calculate
    base_line_spacing = None  # Auto-calculate
    base_num_lines = None  # Auto-calculate

    # Colors
    line_color = (0, 0, 0)  # Black
    bg_color = (255, 255, 255)  # White

    # Generate for each resolution
    resolutions = [
        (64, 1.0),
        (32, 0.5),
        (16, 0.25),
    ]

    for size, scale in resolutions:
        print(f"Generating {size}x{size} image...")

        # Scale thickness (minimum 1 pixel)
        thickness = max(1, int(base_thickness * scale))

        # Generate image
        image = generate_diagonal_pattern(
            size=size,
            line_color=line_color,
            bg_color=bg_color,
            line_thickness=thickness,
            line_length=base_line_length * scale if base_line_length else None,
            line_spacing=base_line_spacing * scale if base_line_spacing else None,
            num_lines=base_num_lines,
            global_angle_deg=45.0,
            segment_angle_deg=-45.0,
            margin_ratio=0.15,
            supersample=12 if size <= 32 else 8,
        )

        # Save image
        filename = f"diagonal_lines_{size}x{size}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"  Saved: {filepath}")

    print(f"\nGeneration complete! Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
