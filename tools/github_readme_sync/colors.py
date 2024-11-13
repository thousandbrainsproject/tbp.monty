# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import sys

from colorama import Fore, Style, init

# Initialize colorama with autoreset to avoid manual resets
init(autoreset=True)


# Function to check if the terminal supports color
def _supports_color():
    if os.getenv("CI", "false").lower() == "true":
        # Disable color in CI environments
        return False
    elif sys.platform == "win32":
        return True  # Colorama handles color support on Windows
    elif hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        return True  # Non-Windows platforms with a TTY
    return False


# Set color constants based on whether color is supported
if _supports_color():
    RED = Fore.RED
    GRAY = Fore.LIGHTBLACK_EX
    GREEN = Fore.GREEN
    WHITE = Fore.WHITE
    CYAN = Fore.CYAN
    BLUE = Fore.BLUE
    RESET = Style.RESET_ALL
else:
    # If color is not supported, use empty strings
    RED = ""
    GRAY = ""
    GREEN = ""
    WHITE = ""
    CYAN = ""
    BLUE = ""
    RESET = ""

# No need to call any functions; the file automatically initializes color support.
