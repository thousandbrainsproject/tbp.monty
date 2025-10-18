# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""There are 4 levels of difficulty for testing compositional models.

Levels of Difficulty for Testing Compositional Models:
1. Logos on flat surfaces
2. Logos on curved surfaces
3. Logos at different rotations
4. Logos with bends
"""

ANGLES = ["030_angles_yb"]

LOGOS = ["021_logo_tbp", "022_logo_numenta"]

FLAT_OBJECTS_WITHOUT_LOGOS = ["001_cube", "006_disk"]

OBJECTS_WITH_LOGOS_LVL1 = [
    "002_cube_tbp_horz",
    "004_cube_numenta_horz",
    "007_disk_tbp_horz",
    "009_disk_numenta_horz",
]

CURVED_OBJECTS_WITHOUT_LOGOS = ["011_cylinder", "016_sphere", "023_mug"]

ALL_PART_OBJECTS = LOGOS + FLAT_OBJECTS_WITHOUT_LOGOS + CURVED_OBJECTS_WITHOUT_LOGOS

OBJECTS_WITH_LOGOS_LVL2 = OBJECTS_WITH_LOGOS_LVL1 + [
    "012_cylinder_tbp_horz",
    "014_cylinder_numenta_horz",
    "017_sphere_tbp_horz",
    "019_sphere_numenta_horz",
    "024_mug_tbp_horz",
    "026_mug_numenta_horz",
]

OBJECTS_WITH_LOGOS_LVL3 = OBJECTS_WITH_LOGOS_LVL2 + [
    "003_cube_tbp_vert",
    "005_cube_numenta_vert",
    "008_disk_tbp_vert",
    "010_disk_numenta_vert",
    "013_cylinder_tbp_vert",
    "015_cylinder_numenta_vert",
    "018_sphere_tbp_vert",
    "020_sphere_numenta_vert",
    "025_mug_tbp_vert",
    "027_mug_numenta_vert",
]

OBJECTS_WITH_LOGOS_LVL4 = OBJECTS_WITH_LOGOS_LVL3 + ["028_mug_tbp_horz_bent"]

PARENT_TO_CHILD_MAPPING = {
    "001_cube": ["001_cube"],
    "002_cube_tbp_horz": ["001_cube", "021_logo_tbp"],
    "003_cube_tbp_vert": ["001_cube", "021_logo_tbp"],
    "004_cube_numenta_horz": ["001_cube", "022_logo_numenta"],
    "005_cube_numenta_vert": ["001_cube", "022_logo_numenta"],
    "006_disk": ["006_disk"],
    "007_disk_tbp_horz": ["006_disk", "021_logo_tbp"],
    "008_disk_tbp_vert": ["006_disk", "021_logo_tbp"],
    "009_disk_numenta_horz": ["006_disk", "022_logo_numenta"],
    "010_disk_numenta_vert": ["006_disk", "022_logo_numenta"],
    "011_cylinder": ["011_cylinder"],
    "012_cylinder_tbp_horz": ["011_cylinder", "021_logo_tbp"],
    "013_cylinder_tbp_vert": ["011_cylinder", "021_logo_tbp"],
    "014_cylinder_numenta_horz": ["011_cylinder", "022_logo_numenta"],
    "015_cylinder_numenta_vert": ["011_cylinder", "022_logo_numenta"],
    "016_sphere": ["016_sphere"],
    "017_sphere_tbp_horz": ["016_sphere", "021_logo_tbp"],
    "018_sphere_tbp_vert": ["016_sphere", "021_logo_tbp"],
    "019_sphere_numenta_horz": ["016_sphere", "022_logo_numenta"],
    "020_sphere_numenta_vert": ["016_sphere", "022_logo_numenta"],
    "021_logo_tbp": ["021_logo_tbp"],
    "022_logo_numenta": ["022_logo_numenta"],
    "023_mug": ["023_mug"],
    "024_mug_tbp_horz": ["023_mug", "021_logo_tbp"],
    "025_mug_tbp_vert": ["023_mug", "021_logo_tbp"],
    "026_mug_numenta_horz": ["023_mug", "022_logo_numenta"],
    "027_mug_numenta_vert": ["023_mug", "022_logo_numenta"],
    "028_mug_tbp_horz_bent": ["023_mug", "021_logo_tbp"],
}
