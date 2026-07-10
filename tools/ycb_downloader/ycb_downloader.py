# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePath
from typing import TypedDict
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import trimesh
from vedo import Mesh

BASE_OBJECT_URL = "https://ycb-benchmarks.s3.amazonaws.com/data"
OBJECT_TYPE = "google_16k"

# This count is based off the number of object directories we had
# with the Habitat version of the YCB dataset.
EXPECTED_OBJECT_COUNT = 79


class ObjectsJsonSchema(TypedDict):
    """A schema for the `objects.json` file.

    This is to help the type checker. We only care about the "objects" key.
    """

    objects: list[str]


def main() -> None:
    """Main function for YCB objects downloader.

    Downloads the YCB object dataset based on the list in their `objects.json`
    manifest file. Processes each object, adding our `metadata.json` to it
    to track the reference position and rotation that MuJoCo needs to position
    the objects correctly.
    """
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    object_names = fetch_object_names()
    objects = 0
    for name in object_names:
        object_dir = output_dir / f"{name[4:]}"
        object_dir.mkdir(parents=True, exist_ok=True)
        if process_object(name, object_dir):
            objects += 1

    # Sanity check that we're getting the expected number of objects
    if objects != EXPECTED_OBJECT_COUNT:
        print(
            f"WARNING: downloader didn't process the expected number "
            f"of objects: {objects}/{EXPECTED_OBJECT_COUNT}\n"
            "Check skipped objects in the output log."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="~/tbp/data/mujoco",
    )
    return parser.parse_args()


def fetch_object_names() -> list[str]:
    """Fetch the names of the objects from the YCB dataset.

    Returns:
        list[str]: A list of YCB objects names.
    """
    with urlopen(f"{BASE_OBJECT_URL}/objects.json") as response:  # noqa: S310
        html = response.read()
        objects: ObjectsJsonSchema = json.loads(html)
    return objects["objects"]


def process_object(name: str, output_dir: Path) -> bool:
    """Process a single object.

    This function downloads and extracts the tarball from the YCB website
    and creates the metadata.json file we need.

    Raises:
        HTTPError: for any errors downloading the tarball other than 404.

    Returns:
        bool: True if object was processed, False if it was skipped.
    """
    # Note: For some reason, the objects.json lists the name of object 27
    # as `027-skillet` but the tarball filename uses `027_skillet`, so
    # we need to special case this one object's name.
    if name == "027-skillet":
        name = "027_skillet"

    download_url = f"{BASE_OBJECT_URL}/google/{name}_{OBJECT_TYPE}.tgz"
    print(f"Downloading {download_url}...")
    try:
        with urlopen(download_url) as response:  # noqa: S310
            with tarfile.open(fileobj=response, mode="r|gz") as tarball:
                print(f"Extracting {name}...")
                for file in tarball:
                    # The filenames in the tarball are two directories deep,
                    # and we want to rename them to remove the number from the
                    # directory name.
                    parts = PurePath(file.name).parts
                    file.name = str(PurePath(*parts[2:]))
                    tarball.extract(file, output_dir)
                print(f"Creating metadata for {name}...")
                create_metadata(output_dir)
    except HTTPError as e:
        if e.code == 404:
            print(f"Tarball not found. Skipping {name}...")
            output_dir.rmdir()
            return False
        raise
    except FileNotFoundError:
        # One of the files is missing a "textured.obj" file, so we're skipping it.
        print(f"Couldn't load model. Skipping {name}...")
        output_dir.rmdir()
        return False
    else:
        return True


def create_metadata(output_dir: Path) -> None:
    """Create the metadata.json file we need.

    The default positioning for the objects when loaded in MuJoCo are
    not centered in the middle of the object in the way that they are when
    loaded in Habitat. We need to calculate a better position and save
    the offset into a metadata file that the MuJoCo simulator will use to
    correctly position the objects.
    """
    with (output_dir / "textured.obj").open("r") as mesh_file:
        mesh = trimesh.load_mesh(
            mesh_file,
            file_type="obj",
        )
    vispy_mesh = Mesh([mesh.vertices, mesh.faces])

    # The names "refpos" and "refquat" come from MuJoCo's names for these fields
    # in mesh definitions. They are short for "reference position" and "reference
    # quaternion", respectively, which define the origin and initial orientation
    # of an object mesh.
    refpos = np.mean(vispy_mesh.bounds().reshape(3, 2), axis=1).tolist()
    # This quaternion represents a 90-degree rotation about the positive-x axis
    # assuming the right-hand rule.
    refquat = [np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0]
    md = {
        "refpos": refpos,
        "refquat": refquat,
    }
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(md, f)


if __name__ == "__main__":
    main()
