"""Allow ``python -m hand_interaction serve``."""

from __future__ import annotations

import sys

from hand_interaction.cli import main

if __name__ == "__main__":
    sys.exit(main())
