# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Script to generate synthetic test images for comparing edge detection methods.

Generates three test suites:
1. Vertical edges at center with varying thicknesses
2. Vertical edges at different off-center positions
3. Vertical edges with distracting lines

All images are 64x64 RGB with both high and low contrast versions.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration Constants (easily modifiable)
# ============================================================================

# Image dimensions
IMAGE_SIZE = 64
CENTER_X = IMAGE_SIZE // 2  # 32
CENTER_Y = IMAGE_SIZE // 2  # 32

# Contrast levels
HIGH_CONTRAST_BG = (0, 0, 0)  # Black background
HIGH_CONTRAST_EDGE = (255, 255, 255)  # White edge

LOW_CONTRAST_BG = (64, 64, 64)  # Dark gray background
LOW_CONTRAST_EDGE = (192, 192, 192)  # Light gray edge

# Test Suite 1: Thickness values to test (in pixels)
THICKNESSES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 24, 28, 32]

# Test Suite 2: Offset values to test (in pixels from center)
OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
OFFSET_EDGE_THICKNESS = 2  # Fixed thickness for offset test suite

# Test Suite 3: Distraction configurations
# Horizontal line positions (y coordinates)
HORIZONTAL_LINE_POSITIONS = [10, 20, 30, 40, 50]
# Diagonal line angles (in degrees, converted to radians)
DIAGONAL_ANGLES = [45, -45, 30, -30, 60, -60]
DISTRACTION_THICKNESS = 2  # Thickness for distraction lines
MAIN_EDGE_THICKNESS = 2  # Thickness for main edge in distraction test suite

# Additional distraction parameters
RANDOM_CLUTTER_NUM_LINES = 20  # Number of random line segments
RANDOM_CLUTTER_MAX_LEN = 15  # Maximum length of random lines
STRIPE_WIDTH = 4  # Width of stripes for textured backgrounds
CHECKERBOARD_CELL_SIZE = 4  # Cell size for checkerboard pattern
GAUSSIAN_BLOB_SIGMA = 5  # Sigma for Gaussian blobs
GAUSSIAN_BLOB_PEAK_DELTA = 80  # Peak intensity delta for blobs
GRADIENT_AMPLITUDE = 80  # Amplitude for illumination gradients
NOISE_SIGMA = 20  # Standard deviation for Gaussian noise
SALT_PEPPER_PROB = 0.02  # Probability for salt-and-pepper noise
GAP_Y_RANGE = (20, 44)  # Y range for gapped edges
WIGGLE_MAX_JITTER = 2  # Maximum jitter for wiggly edges
CIRCLE_RADIUS = 10  # Radius for circle shapes

# Test Suite 4: Angled lines through center
ANGLES = list(range(0, 181, 2))  # 0 to 180 degrees in 2-degree increments

# Test Suites 5 & 6: Distraction offsets
DISTRACTION_OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

# Thickness variations for distraction tests
THICKNESS_VARIATIONS = [2, 4, 8, 14]

# Test Suite 7: Random line intersections
RANDOM_INTERSECTIONS_COUNT = 20

# Test Suite 8: Angled intersections
ANGLED_INTERSECTION_OFFSETS = [0, 2, 4, 8, 16, 24]  # Y offset below center for intersection
ANGLED_INTERSECTION_ANGLES = list(range(0, 166, 15))  # 0 to 165 in 15-deg increments
ANGLED_INTERSECTION_THICKNESSES = [2, 4, 8, 16]
ANGLED_INTERSECTION_MAIN_THICKNESS = 2  # Fixed main vertical line thickness

# Output directory
OUTPUT_DIR = Path("results/synthetic_edge_test_images")


# ============================================================================
# Helper Functions
# ============================================================================


def create_image(background_color: Tuple[int, int, int]) -> np.ndarray:
    """Create a base 64x64x3 RGB image with specified background color.

    Args:
        background_color: RGB tuple (R, G, B) for background color

    Returns:
        64x64x3 numpy array of uint8
    """
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[:, :] = background_color
    return image


def create_vertical_edge(
    image: np.ndarray, x_position: int, thickness: int, color: Tuple[int, int, int]
) -> np.ndarray:
    """Draw a vertical edge on the image.

    Args:
        image: RGB image array to modify
        x_position: X coordinate of edge center
        thickness: Thickness of edge in pixels
        color: RGB tuple (R, G, B) for edge color

    Returns:
        Modified image array
    """
    half_thickness = thickness // 2
    x_start = max(0, x_position - half_thickness)
    x_end = min(IMAGE_SIZE, x_position + half_thickness + (thickness % 2))
    image[:, x_start:x_end] = color
    return image


def create_horizontal_line(
    image: np.ndarray, y_position: int, thickness: int, color: Tuple[int, int, int]
) -> np.ndarray:
    """Draw a horizontal line on the image.

    Args:
        image: RGB image array to modify
        y_position: Y coordinate of line center
        thickness: Thickness of line in pixels
        color: RGB tuple (R, G, B) for line color

    Returns:
        Modified image array
    """
    half_thickness = thickness // 2
    y_start = max(0, y_position - half_thickness)
    y_end = min(IMAGE_SIZE, y_position + half_thickness + (thickness % 2))
    image[y_start:y_end, :] = color
    return image


def create_diagonal_line(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    thickness: int,
    color: Tuple[int, int, int],
) -> np.ndarray:
    """Draw a diagonal line on the image using OpenCV.

    Args:
        image: RGB image array to modify
        start: (x, y) start coordinates
        end: (x, y) end coordinates
        thickness: Thickness of line in pixels
        color: RGB tuple (R, G, B) for line color (OpenCV uses BGR, so we'll convert)

    Returns:
        Modified image array
    """
    # OpenCV uses BGR format
    bgr_color = (color[2], color[1], color[0])
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.line(image_bgr, start, end, bgr_color, thickness)
    # Convert back to RGB
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# ============================================================================
# Distraction Helper Functions
# ============================================================================


def add_random_line_clutter(
    image: np.ndarray,
    num_lines: int = RANDOM_CLUTTER_NUM_LINES,
    max_len: int = RANDOM_CLUTTER_MAX_LEN,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add random line segments as clutter.

    Args:
        image: RGB image array to modify
        num_lines: Number of random line segments to add
        max_len: Maximum length of line segments
        color: RGB tuple for line color

    Returns:
        Modified image array
    """
    h, w, _ = image.shape
    for _ in range(num_lines):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(3, max_len)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)
        image = create_diagonal_line(image, (x1, y1), (x2, y2), 1, color)
    return image


def add_vertical_stripes(
    image: np.ndarray,
    stripe_width: int = STRIPE_WIDTH,
    color_delta: int = 40,
) -> np.ndarray:
    """Add vertical stripes pattern.

    Args:
        image: RGB image array to modify
        stripe_width: Width of each stripe
        color_delta: Intensity change for stripes

    Returns:
        Modified image array
    """
    base = image.copy().astype(np.int16)
    for x in range(0, IMAGE_SIZE, 2 * stripe_width):
        base[:, x : x + stripe_width, :] += color_delta
    return np.clip(base, 0, 255).astype(np.uint8)


def add_horizontal_stripes(
    image: np.ndarray,
    stripe_width: int = STRIPE_WIDTH,
    color_delta: int = 40,
) -> np.ndarray:
    """Add horizontal stripes pattern.

    Args:
        image: RGB image array to modify
        stripe_width: Width of each stripe
        color_delta: Intensity change for stripes

    Returns:
        Modified image array
    """
    base = image.copy().astype(np.int16)
    for y in range(0, IMAGE_SIZE, 2 * stripe_width):
        base[y : y + stripe_width, :, :] += color_delta
    return np.clip(base, 0, 255).astype(np.uint8)


def add_checkerboard(
    image: np.ndarray,
    cell_size: int = CHECKERBOARD_CELL_SIZE,
    color_delta: int = 40,
) -> np.ndarray:
    """Add checkerboard pattern.

    Args:
        image: RGB image array to modify
        cell_size: Size of each checkerboard cell
        color_delta: Intensity change for checkerboard

    Returns:
        Modified image array
    """
    base = image.copy().astype(np.int16)
    for y in range(0, IMAGE_SIZE, cell_size):
        for x in range(0, IMAGE_SIZE, cell_size):
            if ((x // cell_size) + (y // cell_size)) % 2 == 0:
                base[y : y + cell_size, x : x + cell_size, :] += color_delta
    return np.clip(base, 0, 255).astype(np.uint8)


def add_t_junction(
    image: np.ndarray,
    x_edge: int = CENTER_X,
    y_pos: int = CENTER_Y,
    length: int = 20,
    thickness: int = DISTRACTION_THICKNESS,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add a T-junction (horizontal line terminating at vertical edge).

    Args:
        image: RGB image array to modify
        x_edge: X position of the vertical edge
        y_pos: Y position of the horizontal line
        length: Length of horizontal line
        thickness: Thickness of the line
        color: RGB tuple for line color

    Returns:
        Modified image array
    """
    # Horizontal line that terminates at the main edge
    x_start = max(0, x_edge - length)
    x_end = x_edge
    half_t = thickness // 2
    y_start = max(0, y_pos - half_t)
    y_end = min(IMAGE_SIZE, y_pos + half_t + (thickness % 2))
    image[y_start:y_end, x_start:x_end] = color
    return image


def add_l_junction(
    image: np.ndarray,
    x_pos: int = CENTER_X,
    y_pos: int = CENTER_Y,
    length: int = 20,
    thickness: int = DISTRACTION_THICKNESS,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add an L-junction (corner shape).

    Args:
        image: RGB image array to modify
        x_pos: X position of the corner
        y_pos: Y position of the corner
        length: Length of each leg
        thickness: Thickness of the lines
        color: RGB tuple for line color

    Returns:
        Modified image array
    """
    # Vertical leg
    half_t = thickness // 2
    y_start = max(0, y_pos - half_t)
    y_end = min(IMAGE_SIZE, y_pos + half_t + (thickness % 2))
    x_start = max(0, x_pos - half_t)
    x_end = min(IMAGE_SIZE, x_pos + half_t + (thickness % 2))
    image[y_start:y_end, x_start:x_end] = color

    # Horizontal leg
    x_start = max(0, x_pos - half_t)
    x_end = min(IMAGE_SIZE, x_pos + length)
    y_start = max(0, y_pos - half_t)
    y_end = min(IMAGE_SIZE, y_pos + half_t + (thickness % 2))
    image[y_start:y_end, x_start:x_end] = color
    return image


def add_x_junction(
    image: np.ndarray,
    center: Tuple[int, int] = (CENTER_X, CENTER_Y),
    length: int = 15,
    thickness: int = DISTRACTION_THICKNESS,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add an X-junction (crossing lines).

    Args:
        image: RGB image array to modify
        center: (x, y) center position
        length: Length of each line segment
        thickness: Thickness of the lines
        color: RGB tuple for line color

    Returns:
        Modified image array
    """
    cx, cy = center
    # Diagonal line from top-left to bottom-right
    start1 = (max(0, cx - length), max(0, cy - length))
    end1 = (min(IMAGE_SIZE - 1, cx + length), min(IMAGE_SIZE - 1, cy + length))
    image = create_diagonal_line(image, start1, end1, thickness, color)

    # Diagonal line from top-right to bottom-left
    start2 = (min(IMAGE_SIZE - 1, cx + length), max(0, cy - length))
    end2 = (max(0, cx - length), min(IMAGE_SIZE - 1, cy + length))
    return create_diagonal_line(image, start2, end2, thickness, color)


def add_gaussian_blob(
    image: np.ndarray,
    center: Tuple[int, int],
    sigma: float = GAUSSIAN_BLOB_SIGMA,
    peak_delta: int = GAUSSIAN_BLOB_PEAK_DELTA,
) -> np.ndarray:
    """Add a Gaussian blob (bright/dark spot with smoothed boundaries).

    Args:
        image: RGB image array to modify
        center: (x, y) center position of blob
        sigma: Standard deviation of Gaussian
        peak_delta: Peak intensity change

    Returns:
        Modified image array
    """
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    cx, cy = center
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    blob = peak_delta * np.exp(-dist2 / (2 * sigma**2))
    base = image.astype(np.float32)
    for c in range(3):
        base[:, :, c] += blob
    return np.clip(base, 0, 255).astype(np.uint8)


def add_horizontal_gradient(
    image: np.ndarray, amplitude: int = GRADIENT_AMPLITUDE
) -> np.ndarray:
    """Add a horizontal illumination gradient.

    Args:
        image: RGB image array to modify
        amplitude: Amplitude of gradient

    Returns:
        Modified image array
    """
    grad = np.linspace(-amplitude / 2, amplitude / 2, IMAGE_SIZE, dtype=np.float32)
    base = image.astype(np.float32)
    base += grad[None, :, None]
    return np.clip(base, 0, 255).astype(np.uint8)


def add_vertical_gradient(
    image: np.ndarray, amplitude: int = GRADIENT_AMPLITUDE
) -> np.ndarray:
    """Add a vertical illumination gradient.

    Args:
        image: RGB image array to modify
        amplitude: Amplitude of gradient

    Returns:
        Modified image array
    """
    grad = np.linspace(-amplitude / 2, amplitude / 2, IMAGE_SIZE, dtype=np.float32)
    base = image.astype(np.float32)
    base += grad[:, None, None]
    return np.clip(base, 0, 255).astype(np.uint8)


def add_gaussian_noise(
    image: np.ndarray, sigma: float = NOISE_SIGMA
) -> np.ndarray:
    """Add Gaussian noise.

    Args:
        image: RGB image array to modify
        sigma: Standard deviation of noise

    Returns:
        Modified image array
    """
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(
    image: np.ndarray, prob: float = SALT_PEPPER_PROB
) -> np.ndarray:
    """Add salt-and-pepper noise.

    Args:
        image: RGB image array to modify
        prob: Probability of noise pixels

    Returns:
        Modified image array
    """
    noisy = image.copy()
    mask = np.random.rand(*image.shape[:2])
    noisy[mask < prob / 2] = (0, 0, 0)
    noisy[(mask >= prob / 2) & (mask < prob)] = (255, 255, 255)
    return noisy


def create_gapped_vertical_edge(
    image: np.ndarray,
    x_position: int,
    thickness: int,
    color: Tuple[int, int, int],
    gap_y_range: Tuple[int, int] = GAP_Y_RANGE,
) -> np.ndarray:
    """Create a vertical edge with a gap.

    Args:
        image: RGB image array to modify
        x_position: X coordinate of edge center
        thickness: Thickness of edge
        color: RGB tuple for edge color
        gap_y_range: (y_start, y_end) range where gap should be

    Returns:
        Modified image array
    """
    image = create_vertical_edge(image, x_position, thickness, color)
    y_start, y_end = gap_y_range
    half_thickness = thickness // 2
    x_start = max(0, x_position - half_thickness)
    x_end = min(IMAGE_SIZE, x_position + half_thickness + (thickness % 2))
    # Reset gap region to background color (use first pixel as reference)
    bg_color = tuple(image[0, 0])
    image[y_start:y_end, x_start:x_end] = bg_color
    return image


def create_wiggly_vertical_edge(
    image: np.ndarray,
    base_x: int,
    thickness: int,
    color: Tuple[int, int, int],
    max_jitter: int = WIGGLE_MAX_JITTER,
) -> np.ndarray:
    """Create a wiggly vertical edge with random jitter per row.

    Args:
        image: RGB image array to modify
        base_x: Base X position of edge
        thickness: Thickness of edge
        color: RGB tuple for edge color
        max_jitter: Maximum jitter in pixels

    Returns:
        Modified image array
    """
    for y in range(IMAGE_SIZE):
        jitter = np.random.randint(-max_jitter, max_jitter + 1)
        x_pos = np.clip(base_x + jitter, 0, IMAGE_SIZE - 1)
        half_thickness = thickness // 2
        x_start = max(0, x_pos - half_thickness)
        x_end = min(IMAGE_SIZE, x_pos + half_thickness + (thickness % 2))
        image[y, x_start:x_end] = color
    return image


def add_circle(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int = CIRCLE_RADIUS,
    thickness: int = DISTRACTION_THICKNESS,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add a circle shape.

    Args:
        image: RGB image array to modify
        center: (x, y) center position
        radius: Radius of circle
        thickness: Thickness of circle outline
        color: RGB tuple for circle color

    Returns:
        Modified image array
    """
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bgr_color = (color[2], color[1], color[0])
    cv2.circle(img_bgr, center, radius, bgr_color, thickness)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def add_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    thickness: int = DISTRACTION_THICKNESS,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add a rectangle shape.

    Args:
        image: RGB image array to modify
        top_left: (x, y) top-left corner
        bottom_right: (x, y) bottom-right corner
        thickness: Thickness of rectangle outline
        color: RGB tuple for rectangle color

    Returns:
        Modified image array
    """
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bgr_color = (color[2], color[1], color[0])
    cv2.rectangle(img_bgr, top_left, bottom_right, bgr_color, thickness)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def add_triangle(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    pt3: Tuple[int, int],
    thickness: int = DISTRACTION_THICKNESS,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add a triangle shape.

    Args:
        image: RGB image array to modify
        pt1: First vertex (x, y)
        pt2: Second vertex (x, y)
        pt3: Third vertex (x, y)
        thickness: Thickness of triangle outline
        color: RGB tuple for triangle color

    Returns:
        Modified image array
    """
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bgr_color = (color[2], color[1], color[0])
    pts = np.array([pt1, pt2, pt3], np.int32)
    cv2.polylines(img_bgr, [pts], True, bgr_color, thickness)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ============================================================================
# Input Parsing and Validation Functions
# ============================================================================


def parse_suite_selection(input_str: str) -> List[int]:
    """Parse and validate suite selection input.

    Args:
        input_str: Comma or space-separated string of suite numbers
            (e.g., "1,2,3" or "1 2 3")

    Returns:
        List of valid suite numbers (1-8)

    Raises:
        ValueError: If input contains invalid suite numbers or non-numeric values
    """
    # Replace commas with spaces and split
    input_str = input_str.replace(",", " ").strip()
    if not input_str:
        raise ValueError("Empty input provided")

    # Split by spaces and convert to integers
    parts = input_str.split()
    suites = []
    valid_suites = {1, 2, 3, 4, 5, 6, 7, 8}

    for part in parts:
        try:
            suite_num = int(part)
            if suite_num not in valid_suites:
                raise ValueError(
                    f"Invalid suite number: {suite_num}. Must be 1-8."
                )
            suites.append(suite_num)
        except ValueError as e:
            if "Invalid suite number" in str(e):
                raise
            raise ValueError(
                f"Invalid input: '{part}'. Must be a number (1-8)."
            ) from None

    # Remove duplicates while preserving order
    seen = set()
    unique_suites = []
    for suite in suites:
        if suite not in seen:
            seen.add(suite)
            unique_suites.append(suite)

    return unique_suites


def get_suite_selection_interactive() -> List[int]:
    """Get suite selection from user via interactive prompt.

    Returns:
        List of valid suite numbers (1-8)
    """
    print("\nSelect test suites to generate:")
    print("  1 = Thickness variations")
    print("  2 = Offset variations")
    print("  3 = Distractions")
    print("  4 = Angled lines through center")
    print("  5 = Horizontal distraction with offsets")
    print("  6 = Vertical distraction with offsets")
    print("  7 = Random line intersections")
    print("  8 = Angled intersections (vertical line + angled line)")
    print(
        "\nEnter suite numbers (comma or space separated, "
        "e.g., 1,2,3 or 1 2 3):"
    )

    while True:
        try:
            user_input = input("> ").strip()
            return parse_suite_selection(user_input)
        except ValueError as e:
            print(f"Error: {e}")
            print("Please enter valid suite numbers (1-8):")


# ============================================================================
# Test Suite Generation Functions
# ============================================================================


def generate_thickness_test_images(
    output_dir: Path, contrast_level: str
) -> None:
    """Generate test suite 1: vertical edges at center with varying thicknesses.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    for thickness in THICKNESSES:
        # Create image with background
        image = create_image(bg_color)

        # Draw vertical edge at center
        create_vertical_edge(image, CENTER_X, thickness, edge_color)

        # Save image
        filename = f"thickness_{thickness}px_center_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")


def generate_offset_test_images(output_dir: Path, contrast_level: str) -> None:
    """Generate test suite 2: vertical edges at different off-center positions.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    for offset in OFFSETS:
        # Create image with background
        image = create_image(bg_color)

        # Calculate edge position: center + offset
        # Offset can be positive (right) or we'll use absolute value
        x_position = CENTER_X + offset

        # Only generate if edge is within image bounds
        if 0 <= x_position < IMAGE_SIZE:
            # Draw vertical edge at offset position
            create_vertical_edge(image, x_position, OFFSET_EDGE_THICKNESS, edge_color)

            # Save image
            filename = (
                f"offset_{offset}px_thickness_{OFFSET_EDGE_THICKNESS}px_"
                f"{contrast_level}.png"
            )
            filepath = output_dir / filename
            plt.imsave(filepath, image)
            print(f"Generated: {filepath}")


def generate_distraction_test_images(
    output_dir: Path, contrast_level: str
) -> None:
    """Generate test suite 3: vertical edge at center with distracting lines.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    # Generate horizontal line distractions
    for y_pos in HORIZONTAL_LINE_POSITIONS:
        image = create_image(bg_color)
        # Draw main vertical edge at center
        create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
        # Draw horizontal distraction line
        create_horizontal_line(
            image, y_pos, DISTRACTION_THICKNESS, edge_color
        )
        filename = f"distraction_horizontal_y{y_pos}_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate diagonal line distractions
    for angle_deg in DIAGONAL_ANGLES:
        image = create_image(bg_color)
        # Draw main vertical edge at center
        create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)

        # Draw diagonal line across image
        # Convert angle to radians (angle from horizontal, positive = counterclockwise)
        angle_rad = np.deg2rad(angle_deg)

        # Calculate line that spans across the image through center
        # We want a line that goes from one edge to the opposite edge
        # For a line through center with angle theta, we need to find intersection
        # with image boundaries

        # Use a large length to ensure we span the image
        # Calculate direction vector
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        # Find intersections with image boundaries
        # Image boundaries: x in [0, IMAGE_SIZE-1], y in [0, IMAGE_SIZE-1]
        # Line through center: (x, y) = (CENTER_X, CENTER_Y) + t * (dx, dy)

        # Find t values where line intersects boundaries
        t_values = []

        # Intersection with left edge (x = 0)
        if abs(dx) > 1e-6:
            t = (0 - CENTER_X) / dx
            y = CENTER_Y + t * dy
            if 0 <= y < IMAGE_SIZE:
                t_values.append(t)

        # Intersection with right edge (x = IMAGE_SIZE - 1)
        if abs(dx) > 1e-6:
            t = (IMAGE_SIZE - 1 - CENTER_X) / dx
            y = CENTER_Y + t * dy
            if 0 <= y < IMAGE_SIZE:
                t_values.append(t)

        # Intersection with top edge (y = 0)
        if abs(dy) > 1e-6:
            t = (0 - CENTER_Y) / dy
            x = CENTER_X + t * dx
            if 0 <= x < IMAGE_SIZE:
                t_values.append(t)

        # Intersection with bottom edge (y = IMAGE_SIZE - 1)
        if abs(dy) > 1e-6:
            t = (IMAGE_SIZE - 1 - CENTER_Y) / dy
            x = CENTER_X + t * dx
            if 0 <= x < IMAGE_SIZE:
                t_values.append(t)

        # Get the two extreme t values (min and max)
        if len(t_values) >= 2:
            t_min = min(t_values)
            t_max = max(t_values)

            start = (
                int(CENTER_X + t_min * dx),
                int(CENTER_Y + t_min * dy),
            )
            end = (
                int(CENTER_X + t_max * dx),
                int(CENTER_Y + t_max * dy),
            )

            # Clamp to image bounds
            start = (
                max(0, min(IMAGE_SIZE - 1, start[0])),
                max(0, min(IMAGE_SIZE - 1, start[1])),
            )
            end = (
                max(0, min(IMAGE_SIZE - 1, end[0])),
                max(0, min(IMAGE_SIZE - 1, end[1])),
            )

            image = create_diagonal_line(
                image, start, end, DISTRACTION_THICKNESS, edge_color
            )
        elif abs(angle_deg) == 45:
            # Fallback: use corner-to-corner for 45 degree angles
            if angle_deg > 0:
                start = (0, IMAGE_SIZE - 1)
                end = (IMAGE_SIZE - 1, 0)
            else:
                start = (0, 0)
                end = (IMAGE_SIZE - 1, IMAGE_SIZE - 1)
            image = create_diagonal_line(
                image, start, end, DISTRACTION_THICKNESS, edge_color
            )

        filename = f"distraction_diagonal_{angle_deg}deg_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate random line clutter distractions
    for seed in [0, 1, 2]:  # Generate 3 variations
        np.random.seed(seed)
        image = create_image(bg_color)
        create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
        image = add_random_line_clutter(image, color=edge_color)
        filename = f"distraction_clutter_seed{seed}_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate vertical stripes
    image = create_image(bg_color)
    create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
    image = add_vertical_stripes(image)
    filename = f"distraction_vertical_stripes_{contrast_level}.png"
    filepath = output_dir / filename
    plt.imsave(filepath, image)
    print(f"Generated: {filepath}")

    # Generate horizontal stripes
    image = create_image(bg_color)
    create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
    image = add_horizontal_stripes(image)
    filename = f"distraction_horizontal_stripes_{contrast_level}.png"
    filepath = output_dir / filename
    plt.imsave(filepath, image)
    print(f"Generated: {filepath}")

    # Generate checkerboard
    image = create_image(bg_color)
    create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
    image = add_checkerboard(image)
    filename = f"distraction_checkerboard_{contrast_level}.png"
    filepath = output_dir / filename
    plt.imsave(filepath, image)
    print(f"Generated: {filepath}")

    # Generate T-junctions
    for y_pos in [20, 32, 44]:
        image = create_image(bg_color)
        create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
        image = add_t_junction(image, y_pos=y_pos, color=edge_color)
        filename = f"distraction_tjunction_y{y_pos}_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate L-junctions
    for x_pos, y_pos in [(20, 20), (44, 44), (20, 44)]:
        image = create_image(bg_color)
        create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
        image = add_l_junction(image, x_pos=x_pos, y_pos=y_pos, color=edge_color)
        filename = f"distraction_ljunction_x{x_pos}_y{y_pos}_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate X-junctions
    for center in [(20, 20), (CENTER_X, CENTER_Y), (44, 44)]:
        image = create_image(bg_color)
        create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
        image = add_x_junction(image, center=center, color=edge_color)
        filename = (
            f"distraction_xjunction_x{center[0]}_y{center[1]}_"
            f"{contrast_level}.png"
        )
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate Gaussian blobs
    for center in [(16, 16), (48, 16), (16, 48), (48, 48)]:
        image = create_image(bg_color)
        create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
        image = add_gaussian_blob(image, center=center)
        filename = f"distraction_blob_x{center[0]}_y{center[1]}_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate horizontal gradient
    image = create_image(bg_color)
    create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
    image = add_horizontal_gradient(image)
    filename = f"distraction_horizontal_gradient_{contrast_level}.png"
    filepath = output_dir / filename
    plt.imsave(filepath, image)
    print(f"Generated: {filepath}")

    # Generate vertical gradient
    image = create_image(bg_color)
    create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
    image = add_vertical_gradient(image)
    filename = f"distraction_vertical_gradient_{contrast_level}.png"
    filepath = output_dir / filename
    plt.imsave(filepath, image)
    print(f"Generated: {filepath}")

    # Generate Gaussian noise
    for seed in [0, 1, 2]:  # Generate 3 variations
        np.random.seed(seed)
        image = create_image(bg_color)
        create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
        image = add_gaussian_noise(image)
        filename = f"distraction_gaussian_noise_seed{seed}_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate salt-and-pepper noise
    for seed in [0, 1, 2]:  # Generate 3 variations
        np.random.seed(seed)
        image = create_image(bg_color)
        create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
        image = add_salt_pepper_noise(image)
        filename = f"distraction_saltpepper_seed{seed}_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate gapped edges
    for gap_range in [(10, 20), (20, 44), (44, 54)]:
        image = create_image(bg_color)
        image = create_gapped_vertical_edge(
            image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color, gap_y_range=gap_range
        )
        filename = (
            f"distraction_gapped_y{gap_range[0]}_{gap_range[1]}_"
            f"{contrast_level}.png"
        )
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate wiggly edges
    for seed in [0, 1, 2]:  # Generate 3 variations
        np.random.seed(seed)
        image = create_image(bg_color)
        image = create_wiggly_vertical_edge(
            image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color
        )
        filename = f"distraction_wiggly_seed{seed}_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate circles
    for center in [(16, CENTER_Y), (48, CENTER_Y), (CENTER_X, 16), (CENTER_X, 48)]:
        image = create_image(bg_color)
        create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
        image = add_circle(image, center=center, color=edge_color)
        filename = f"distraction_circle_x{center[0]}_y{center[1]}_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate rectangles
    for rect_config in [
        ((10, 10), (22, 22)),
        ((42, 10), (54, 22)),
        ((10, 42), (22, 54)),
        ((42, 42), (54, 54)),
    ]:
        image = create_image(bg_color)
        create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
        image = add_rectangle(image, rect_config[0], rect_config[1], color=edge_color)
        filename = (
            f"distraction_rectangle_{rect_config[0][0]}_{rect_config[0][1]}_"
            f"{contrast_level}.png"
        )
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")

    # Generate triangles
    for tri_config in [
        ((16, 10), (10, 22), (22, 22)),
        ((48, 10), (42, 22), (54, 22)),
        ((16, 42), (10, 54), (22, 54)),
        ((48, 42), (42, 54), (54, 54)),
    ]:
        image = create_image(bg_color)
        create_vertical_edge(image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color)
        image = add_triangle(
            image, tri_config[0], tri_config[1], tri_config[2], color=edge_color
        )
        filename = (
            f"distraction_triangle_{tri_config[0][0]}_{tri_config[0][1]}_"
            f"{contrast_level}.png"
        )
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")


def generate_angled_lines_test_images(
    output_dir: Path, contrast_level: str
) -> None:
    """Generate test suite 4: angled lines through center at different angles.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    for angle_deg in ANGLES:
        # Create image with background
        image = create_image(bg_color)

        # Convert angle to radians (angle from horizontal, positive = counterclockwise)
        angle_rad = np.deg2rad(angle_deg)

        # Calculate direction vector
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        # Find intersections with image boundaries
        # Line through center: (x, y) = (CENTER_X, CENTER_Y) + t * (dx, dy)
        t_values = []

        # Intersection with left edge (x = 0)
        if abs(dx) > 1e-6:
            t = (0 - CENTER_X) / dx
            y = CENTER_Y + t * dy
            if 0 <= y < IMAGE_SIZE:
                t_values.append(t)

        # Intersection with right edge (x = IMAGE_SIZE - 1)
        if abs(dx) > 1e-6:
            t = (IMAGE_SIZE - 1 - CENTER_X) / dx
            y = CENTER_Y + t * dy
            if 0 <= y < IMAGE_SIZE:
                t_values.append(t)

        # Intersection with top edge (y = 0)
        if abs(dy) > 1e-6:
            t = (0 - CENTER_Y) / dy
            x = CENTER_X + t * dx
            if 0 <= x < IMAGE_SIZE:
                t_values.append(t)

        # Intersection with bottom edge (y = IMAGE_SIZE - 1)
        if abs(dy) > 1e-6:
            t = (IMAGE_SIZE - 1 - CENTER_Y) / dy
            x = CENTER_X + t * dx
            if 0 <= x < IMAGE_SIZE:
                t_values.append(t)

        # Get the two extreme t values (min and max)
        if len(t_values) >= 2:
            t_min = min(t_values)
            t_max = max(t_values)

            start = (
                int(CENTER_X + t_min * dx),
                int(CENTER_Y + t_min * dy),
            )
            end = (
                int(CENTER_X + t_max * dx),
                int(CENTER_Y + t_max * dy),
            )

            # Clamp to image bounds
            start = (
                max(0, min(IMAGE_SIZE - 1, start[0])),
                max(0, min(IMAGE_SIZE - 1, start[1])),
            )
            end = (
                max(0, min(IMAGE_SIZE - 1, end[0])),
                max(0, min(IMAGE_SIZE - 1, end[1])),
            )

            image = create_diagonal_line(
                image, start, end, MAIN_EDGE_THICKNESS, edge_color
            )
        elif angle_deg == 0 or angle_deg == 180:
            # Horizontal line
            image = create_horizontal_line(
                image, CENTER_Y, MAIN_EDGE_THICKNESS, edge_color
            )
        elif angle_deg == 90:
            # Vertical line
            image = create_vertical_edge(
                image, CENTER_X, MAIN_EDGE_THICKNESS, edge_color
            )

        # Save image
        filename = f"angled_{angle_deg}deg_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")


def generate_horizontal_distraction_offset_test_images(
    output_dir: Path, contrast_level: str
) -> None:
    """Generate test suite 5: horizontal distraction lines with offsets.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    # Define thickness combinations
    # Same thickness: (main, distraction)
    same_thickness = [(2, 2), (4, 4), (8, 8), (14, 14)]
    # Main thicker: (main, distraction)
    main_thicker = [(4, 2), (8, 2), (8, 4), (14, 2), (14, 4), (14, 8)]
    # Distraction thicker: (main, distraction)
    dist_thicker = [(2, 4), (2, 8), (2, 14), (4, 8), (4, 14), (8, 14)]

    thickness_combinations = same_thickness + main_thicker + dist_thicker

    for offset in DISTRACTION_OFFSETS:
        for main_thick, dist_thick in thickness_combinations:
            # Create image with background
            image = create_image(bg_color)

            # Draw main vertical edge at center
            create_vertical_edge(image, CENTER_X, main_thick, edge_color)

            # Draw horizontal distraction line at center Y, offset by offset pixels
            y_position = CENTER_Y + offset
            # Only generate if line is within image bounds
            if 0 <= y_position < IMAGE_SIZE:
                create_horizontal_line(
                    image, y_position, dist_thick, edge_color
                )

                # Save image
                filename = (
                    f"horizontal_distraction_offset{offset}_"
                    f"main{main_thick}_dist{dist_thick}_{contrast_level}.png"
                )
                filepath = output_dir / filename
                plt.imsave(filepath, image)
                print(f"Generated: {filepath}")


def generate_vertical_distraction_offset_test_images(
    output_dir: Path, contrast_level: str
) -> None:
    """Generate test suite 6: vertical distraction lines with offsets.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    # Define thickness combinations (same as suite 5)
    same_thickness = [(2, 2), (4, 4), (8, 8), (14, 14)]
    main_thicker = [(4, 2), (8, 2), (8, 4), (14, 2), (14, 4), (14, 8)]
    dist_thicker = [(2, 4), (2, 8), (2, 14), (4, 8), (4, 14), (8, 14)]

    thickness_combinations = same_thickness + main_thicker + dist_thicker

    for offset in DISTRACTION_OFFSETS:
        for main_thick, dist_thick in thickness_combinations:
            # Create image with background
            image = create_image(bg_color)

            # Draw main vertical edge at center
            create_vertical_edge(image, CENTER_X, main_thick, edge_color)

            # Draw vertical distraction line at center X, offset by offset pixels
            x_position = CENTER_X + offset
            # Only generate if line is within image bounds
            if 0 <= x_position < IMAGE_SIZE:
                create_vertical_edge(image, x_position, dist_thick, edge_color)

                # Save image
                filename = (
                    f"vertical_distraction_offset{offset}_"
                    f"main{main_thick}_dist{dist_thick}_{contrast_level}.png"
                )
                filepath = output_dir / filename
                plt.imsave(filepath, image)
                print(f"Generated: {filepath}")


def generate_random_intersections_test_images(
    output_dir: Path, contrast_level: str
) -> None:
    """Generate test suite 7: random line intersections with random thickness.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    for seed in range(RANDOM_INTERSECTIONS_COUNT):
        np.random.seed(seed)
        image = create_image(bg_color)

        # Generate 3-5 random lines
        num_lines = np.random.randint(3, 6)

        for _ in range(num_lines):
            # Random thickness: 1-6 pixels
            thickness = np.random.randint(1, 7)

            # Random start point
            x1 = np.random.randint(0, IMAGE_SIZE)
            y1 = np.random.randint(0, IMAGE_SIZE)

            # Random angle
            angle_rad = np.random.uniform(0, 2 * np.pi)

            # Random length
            length = np.random.randint(10, IMAGE_SIZE)

            # Calculate end point
            x2 = int(x1 + length * np.cos(angle_rad))
            y2 = int(y1 + length * np.sin(angle_rad))

            # Clamp to image bounds
            x2 = np.clip(x2, 0, IMAGE_SIZE - 1)
            y2 = np.clip(y2, 0, IMAGE_SIZE - 1)

            # Draw line
            image = create_diagonal_line(
                image, (x1, y1), (x2, y2), thickness, edge_color
            )

        # Save image
        filename = f"random_intersections_seed{seed}_{contrast_level}.png"
        filepath = output_dir / filename
        plt.imsave(filepath, image)
        print(f"Generated: {filepath}")


def generate_angled_intersections_test_images(
    output_dir: Path, contrast_level: str
) -> None:
    """Generate test suite 8: angled line intersections with vertical line.

    Creates images with a main vertical line at center and an intersecting line
    at various angles, with the intersection point offset below center.

    Args:
        output_dir: Directory to save images
        contrast_level: 'high' or 'low'
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if contrast_level == "high":
        bg_color = HIGH_CONTRAST_BG
        edge_color = HIGH_CONTRAST_EDGE
    else:
        bg_color = LOW_CONTRAST_BG
        edge_color = LOW_CONTRAST_EDGE

    for offset in ANGLED_INTERSECTION_OFFSETS:
        # Intersection point is at center X, offset below center Y
        intersection_x = CENTER_X
        intersection_y = CENTER_Y + offset

        for angle_deg in ANGLED_INTERSECTION_ANGLES:
            for thickness in ANGLED_INTERSECTION_THICKNESSES:
                # Create image with background
                image = create_image(bg_color)

                # Draw main vertical edge at center with fixed thickness
                create_vertical_edge(
                    image, CENTER_X, ANGLED_INTERSECTION_MAIN_THICKNESS, edge_color
                )

                # Draw intersecting line at the specified angle through intersection point
                # Convert angle to radians (angle from horizontal)
                angle_rad = np.deg2rad(angle_deg)

                # Calculate direction vector
                dx = np.cos(angle_rad)
                dy = np.sin(angle_rad)

                # Find intersections with image boundaries
                # Line through intersection point: (x, y) = (ix, iy) + t * (dx, dy)
                t_values = []

                # Intersection with left edge (x = 0)
                if abs(dx) > 1e-6:
                    t = (0 - intersection_x) / dx
                    y = intersection_y + t * dy
                    if 0 <= y < IMAGE_SIZE:
                        t_values.append(t)

                # Intersection with right edge (x = IMAGE_SIZE - 1)
                if abs(dx) > 1e-6:
                    t = (IMAGE_SIZE - 1 - intersection_x) / dx
                    y = intersection_y + t * dy
                    if 0 <= y < IMAGE_SIZE:
                        t_values.append(t)

                # Intersection with top edge (y = 0)
                if abs(dy) > 1e-6:
                    t = (0 - intersection_y) / dy
                    x = intersection_x + t * dx
                    if 0 <= x < IMAGE_SIZE:
                        t_values.append(t)

                # Intersection with bottom edge (y = IMAGE_SIZE - 1)
                if abs(dy) > 1e-6:
                    t = (IMAGE_SIZE - 1 - intersection_y) / dy
                    x = intersection_x + t * dx
                    if 0 <= x < IMAGE_SIZE:
                        t_values.append(t)

                # Get the two extreme t values (min and max)
                if len(t_values) >= 2:
                    t_min = min(t_values)
                    t_max = max(t_values)

                    start = (
                        int(intersection_x + t_min * dx),
                        int(intersection_y + t_min * dy),
                    )
                    end = (
                        int(intersection_x + t_max * dx),
                        int(intersection_y + t_max * dy),
                    )

                    # Clamp to image bounds
                    start = (
                        max(0, min(IMAGE_SIZE - 1, start[0])),
                        max(0, min(IMAGE_SIZE - 1, start[1])),
                    )
                    end = (
                        max(0, min(IMAGE_SIZE - 1, end[0])),
                        max(0, min(IMAGE_SIZE - 1, end[1])),
                    )

                    image = create_diagonal_line(
                        image, start, end, thickness, edge_color
                    )
                elif angle_deg == 90:
                    # Special case: vertical line (same as main line, creates overlap)
                    create_vertical_edge(image, intersection_x, thickness, edge_color)

                # Save image
                filename = (
                    f"angled_intersection_offset{offset}_"
                    f"angle{angle_deg}_thick{thickness}_{contrast_level}.png"
                )
                filepath = output_dir / filename
                plt.imsave(filepath, image)
                print(f"Generated: {filepath}")


# ============================================================================
# Main Function
# ============================================================================


def main(suites: List[int] = None):
    """Main function to generate selected test suites.

    Args:
        suites: List of suite numbers to generate
            (1=thickness, 2=offset, 3=distractions, 4=angled lines,
            5=horizontal distraction offsets, 6=vertical distraction offsets,
            7=random intersections, 8=angled intersections).
            If None, will prompt user interactively.
    """
    # Get suite selection if not provided
    if suites is None:
        suites = get_suite_selection_interactive()

    # Define output directories for each suite
    suite_dirs = {
        1: OUTPUT_DIR / "thickness",
        2: OUTPUT_DIR / "offset",
        3: OUTPUT_DIR / "distractions",
        4: OUTPUT_DIR / "angled_lines",
        5: OUTPUT_DIR / "horizontal_distraction_offset",
        6: OUTPUT_DIR / "vertical_distraction_offset",
        7: OUTPUT_DIR / "random_intersections",
        8: OUTPUT_DIR / "angled_intersections",
    }

    # Check if any selected suite's output directory exists
    existing_dirs = [suite_dirs[s] for s in suites if suite_dirs[s].exists()]
    if existing_dirs:
        print("\nThe following output directories already exist:")
        for d in existing_dirs:
            print(f"  - {d.absolute()}")
        response = input("Do you want to override existing files? (y/n): ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborting. No files were modified.")
            return

    print("=" * 60)
    print("Generating synthetic edge detection test images")
    print(f"Selected test suites: {', '.join(map(str, suites))}")
    print("=" * 60)

    # Get output directories
    thickness_dir = suite_dirs[1]
    offset_dir = suite_dirs[2]
    distractions_dir = suite_dirs[3]
    angled_lines_dir = suite_dirs[4]
    horizontal_distraction_offset_dir = suite_dirs[5]
    vertical_distraction_offset_dir = suite_dirs[6]
    random_intersections_dir = suite_dirs[7]
    angled_intersections_dir = suite_dirs[8]

    # Generate test suite 1: Thickness variations
    if 1 in suites:
        print("\nGenerating Test Suite 1: Variable Thickness (High Contrast)...")
        generate_thickness_test_images(thickness_dir, "high")
        print("\nGenerating Test Suite 1: Variable Thickness (Low Contrast)...")
        generate_thickness_test_images(thickness_dir, "low")

    # Generate test suite 2: Offset variations
    if 2 in suites:
        print("\nGenerating Test Suite 2: Off-Center Edges (High Contrast)...")
        generate_offset_test_images(offset_dir, "high")
        print("\nGenerating Test Suite 2: Off-Center Edges (Low Contrast)...")
        generate_offset_test_images(offset_dir, "low")

    # Generate test suite 3: Distractions
    if 3 in suites:
        print("\nGenerating Test Suite 3: Edge with Distractions (High Contrast)...")
        generate_distraction_test_images(distractions_dir, "high")
        print("\nGenerating Test Suite 3: Edge with Distractions (Low Contrast)...")
        generate_distraction_test_images(distractions_dir, "low")

    # Generate test suite 4: Angled lines through center
    if 4 in suites:
        print("\nGenerating Test Suite 4: Angled Lines (High Contrast)...")
        generate_angled_lines_test_images(angled_lines_dir, "high")
        print("\nGenerating Test Suite 4: Angled Lines (Low Contrast)...")
        generate_angled_lines_test_images(angled_lines_dir, "low")

    # Generate test suite 5: Horizontal distraction with offsets
    if 5 in suites:
        print(
            "\nGenerating Test Suite 5: Horizontal Distraction Offsets "
            "(High Contrast)..."
        )
        generate_horizontal_distraction_offset_test_images(
            horizontal_distraction_offset_dir, "high"
        )

    # Generate test suite 6: Vertical distraction with offsets
    if 6 in suites:
        print(
            "\nGenerating Test Suite 6: Vertical Distraction Offsets "
            "(High Contrast)..."
        )
        generate_vertical_distraction_offset_test_images(
            vertical_distraction_offset_dir, "high"
        )

    # Generate test suite 7: Random line intersections
    if 7 in suites:
        print(
            "\nGenerating Test Suite 7: Random Line Intersections "
            "(High Contrast)..."
        )
        generate_random_intersections_test_images(random_intersections_dir, "high")
        print(
            "\nGenerating Test Suite 7: Random Line Intersections "
            "(Low Contrast)..."
        )
        generate_random_intersections_test_images(random_intersections_dir, "low")

    # Generate test suite 8: Angled intersections
    if 8 in suites:
        print(
            "\nGenerating Test Suite 8: Angled Intersections "
            "(High Contrast)..."
        )
        generate_angled_intersections_test_images(angled_intersections_dir, "high")

    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic test images for comparing edge detection methods."
        )
    )
    parser.add_argument(
        "--suite",
        "-s",
        type=str,
        default=None,
        help=(
            "Test suites to generate (comma or space separated). "
            "1=thickness, 2=offset, 3=distractions, 4=angled lines, "
            "5=horizontal distraction offsets, 6=vertical distraction offsets, "
            "7=random intersections, 8=angled intersections. "
            "Example: --suite 1,2,3 or -s 1 2 3"
        ),
    )

    args = parser.parse_args()

    # Parse suite selection from command line if provided
    suites = None
    if args.suite:
        try:
            suites = parse_suite_selection(args.suite)
        except ValueError as e:
            print(f"Error parsing suite selection: {e}")
            print("Please provide valid suite numbers (1-8)")
            sys.exit(1)

    main(suites)

