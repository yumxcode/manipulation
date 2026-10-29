#!/usr/bin/env python3
"""薄壳：实际实现在 dobot_plug.apps.inspect_data（安装包后有 console script `dobot-inspect-data`）。"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dobot_plug.apps.inspect_data import main  # noqa: E402

if __name__ == "__main__":
    main()
