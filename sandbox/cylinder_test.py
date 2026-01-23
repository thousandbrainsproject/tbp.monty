import numpy as np


class SyntheticTwoDPose:
    """Minimal subset of TwoDPoseSM needed for a synthetic test.

    Uses Levi-Civita parallel transport to maintain consistent tangent frame
    orientation as the surface normal changes. Integrates projected 3D
    displacements to get cumulative 2D positions.
    """

    def __init__(self):
        self._previous_location = None
        self._cumulative_2d_position = np.zeros(2)
        self._basis_u = None
        self._basis_v = None
        self._previous_normal = None

    def _initialize_basis(self, surface_normal: np.ndarray) -> None:
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(world_up, surface_normal)) > 0.95:
            world_up = np.array([0.0, 0.0, 1.0])

        u = np.cross(world_up, surface_normal)
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            u = np.array([1.0, 0.0, 0.0])
            u_norm = 1.0

        self._basis_u = u / u_norm
        self._basis_v = np.cross(surface_normal, self._basis_u)
        self._previous_normal = surface_normal

    def _parallel_transport_basis(self, new_normal: np.ndarray) -> None:
        """Parallel transport tangent frame using Levi-Civita connection."""
        old_normal = self._previous_normal
        dot = np.clip(np.dot(old_normal, new_normal), -1.0, 1.0)

        # Nearly parallel normals - no rotation needed
        if dot > 1.0 - 1e-10:
            self._previous_normal = new_normal.copy()
            return

        # Nearly anti-parallel - 180 degree rotation
        if dot < -1.0 + 1e-10:
            self._basis_u = -self._basis_u
            self._basis_v = -self._basis_v
            self._previous_normal = new_normal.copy()
            return

        # General case: rotate around cross product axis using Rodrigues' formula
        axis = np.cross(old_normal, new_normal)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(dot)
        c, s = np.cos(angle), np.sin(angle)

        def rodrigues(v: np.ndarray) -> np.ndarray:
            return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)

        self._basis_u = rodrigues(self._basis_u)
        self._basis_v = rodrigues(self._basis_v)
        self._previous_normal = new_normal.copy()

    def integrate_sequence(self, locations: np.ndarray, normals: np.ndarray):
        """
        Given arrays of surface points and normals (N,3), return 2D positions (N,2)
        using the same integration logic as TwoDPoseSM._update_2d_position_and_displacement
        (without curvature correction).
        """
        uv = []

        for i, (loc, n) in enumerate(zip(locations, normals)):
            if i == 0 or self._previous_location is None:
                self._initialize_basis(n)
                self._previous_location = loc.copy()
                self._cumulative_2d_position = np.zeros(2)
                uv.append(self._cumulative_2d_position.copy())
                continue

            self._parallel_transport_basis(n)
            disp = loc - self._previous_location
            chord_length = np.linalg.norm(disp)
            if chord_length < 1e-12:
                uv.append(self._cumulative_2d_position.copy())
                continue

            du_raw = np.dot(disp, self._basis_u)
            dv_raw = np.dot(disp, self._basis_v)
            direction_uv = np.array([du_raw, dv_raw])
            dir_norm = np.linalg.norm(direction_uv)
            if dir_norm > 1e-12:
                direction_uv /= dir_norm

            step_mag = chord_length
            du, dv = direction_uv * step_mag
            self._cumulative_2d_position += np.array([du, dv])
            self._previous_location = loc.copy()
            uv.append(self._cumulative_2d_position.copy())

        return np.array(uv)


def intersect_cylinder(camera_pos, ray_dir, R):
    """
    Intersect ray p = c + t*d with infinite cylinder x^2+z^2 = R^2.
    Returns (t, hit_point) with smallest t>0, or (None, None) if no hit.
    """
    cx, cy, cz = camera_pos
    dx, dy, dz = ray_dir

    A = dx * dx + dz * dz
    B = 2 * (cx * dx + cz * dz)
    C = cx * cx + cz * cz - R * R

    if abs(A) < 1e-12:
        return None, None

    disc = B * B - 4 * A * C
    if disc < 0:
        return None, None

    sqrt_disc = np.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2 * A)
    t2 = (-B + sqrt_disc) / (2 * A)
    ts = [t for t in (t1, t2) if t > 1e-6]
    if not ts:
        return None, None

    t = min(ts)
    hit = camera_pos + t * ray_dir
    return t, hit


def naive_spiral_scan_points(
    R=0.03,
    H=0.05,
    n_steps=400,
    cam_pos=np.array([0.0, 0.15, 0.25]),
):
    """
    Approximate a NaiveScan-like spiral over the front of the cylinder.

    - Camera at fixed position.
    - Yaw/pitch follow a LURD spiral sequence (up, left, down, right) with growing run lengths.
    - For each pose, cast ray through image center and intersect cylinder.
    """

    def center_ray(yaw, pitch):
        # yaw about Y, pitch about X; initial forward is -Z
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        f = np.array([0.0, 0.0, -1.0])

        Ry = np.array(
            [
                [cy, 0.0, sy],
                [0.0, 1.0, 0.0],
                [-sy, 0.0, cy],
            ]
        )
        Rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cp, -sp],
                [0.0, sp, cp],
            ]
        )
        d = Ry @ (Rx @ f)
        return d / np.linalg.norm(d)

    fixed = np.deg2rad(3.0)  # angular step, roughly similar to a small Look/Turn
    actions = ["up", "left", "down", "right"]

    yaw = 0.0
    pitch = 0.0
    yaw_list = []
    pitch_list = []

    steps_per_action = 1
    total_steps = 0
    while total_steps < n_steps:
        for a in actions:
            for _ in range(steps_per_action):
                if total_steps >= n_steps:
                    break
                if a == "up":
                    pitch += fixed
                elif a == "down":
                    pitch -= fixed
                elif a == "left":
                    yaw += fixed
                elif a == "right":
                    yaw -= fixed

                yaw_list.append(yaw)
                pitch_list.append(pitch)
                total_steps += 1
            if a in ("down", "right"):
                steps_per_action += 1

    locations = []
    normals = []
    for y, p in zip(yaw_list, pitch_list):
        d = center_ray(y, p)
        t, hit = intersect_cylinder(cam_pos, d, R)
        if hit is None:
            continue
        x, yy, z = hit
        # finite cylinder height
        if abs(yy) > H / 2:
            continue
        locations.append(hit)
        # cylinder axis along Y; outward normal is radial in XZ
        n = np.array([x, 0.0, z])
        n /= np.linalg.norm(n)
        normals.append(n)

    return np.array(locations), np.array(normals)


def front_face_band(
    R=0.03,
    H=0.05,
    n_theta=60,
    n_z=20,
    theta_range=np.pi / 4,
):
    """
    Sample only the front 'face' band of the cylinder:

    theta in [-theta_range, +theta_range] (around x>0 side facing the camera),
    z in [-H/2, H/2].
    """
    thetas = np.linspace(-theta_range, theta_range, n_theta)
    zs = np.linspace(-H / 2, H / 2, n_z)

    locs = []
    norms = []
    for z in zs:
        for th in thetas:
            x = R * np.cos(th)
            y = z
            zz = R * np.sin(th)
            locs.append([x, y, zz])
            n = np.array([np.cos(th), 0.0, np.sin(th)])
            norms.append(n)
    return np.array(locs), np.array(norms)


if __name__ == "__main__":
    sm = SyntheticTwoDPose()

    # 1) Naive-scan-like spiral over visible cylinder front
    loc_spiral, n_spiral = naive_spiral_scan_points()
    if loc_spiral.size > 0:
        uv_spiral = sm.integrate_sequence(loc_spiral, n_spiral)
    else:
        uv_spiral = np.zeros((0, 2))

    # 2) Front-face band sampling (rectangular patch in (theta, z))
    loc_band, n_band = front_face_band()
    sm2 = SyntheticTwoDPose()
    uv_band = sm2.integrate_sequence(loc_band, n_band)

    import pandas as pd

    if loc_spiral.size > 0:
        pd.DataFrame(
            {
                "x": loc_spiral[:, 0],
                "y": loc_spiral[:, 1],
                "z": loc_spiral[:, 2],
                "u": uv_spiral[:, 0],
                "v": uv_spiral[:, 1],
            }
        ).to_csv("cyl_spiral_uv.csv", index=False)

    pd.DataFrame(
        {
            "x": loc_band[:, 0],
            "y": loc_band[:, 1],
            "z": loc_band[:, 2],
            "u": uv_band[:, 0],
            "v": uv_band[:, 1],
        }
    ).to_csv("cyl_front_band_uv.csv", index=False)

    print("Spiral points:", loc_spiral.shape)
    print("Band points:  ", loc_band.shape)
