# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import importlib.util
import os

import semver


def get_version():
    version_module_path = os.path.join(
        os.path.dirname(__file__), "../../src/tbp/monty/__init__.py"
    )

    spec = importlib.util.spec_from_file_location("tbp.monty", version_module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.__version__


def parse_version(version, part):
    parsed_version = semver.VersionInfo.parse(version)

    if part == "full" or part == "":
        return str(parsed_version)
    elif part == "major":
        return str(parsed_version.major)
    elif part == "minor":
        return f"{parsed_version.major}.{parsed_version.minor}"
    elif part == "patch":
        return f"{parsed_version.major}.{parsed_version.minor}.{parsed_version.patch}"
    else:
        raise ValueError(f"Unknown version part: {part}")


def main():
    parser = argparse.ArgumentParser(description="Version parser")
    parser.add_argument(
        "part",
        nargs="?",
        choices=["full", "major", "minor", "patch"],
        help="Which part of the version to return",
        default="full",
    )

    args = parser.parse_args()
    version = get_version()

    try:
        print(parse_version(version, args.part))
    except ValueError as e:
        print(str(e))


if __name__ == "__main__":
    main()
