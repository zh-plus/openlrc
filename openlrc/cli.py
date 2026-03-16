#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import subprocess
import sys


def run_gui():
    try:
        # Assuming the Streamlit app is located at "openlrc/gui/home.py"
        subprocess.run(["streamlit", "run", "openlrc/gui/home.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to launch the Streamlit app: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        run_gui()
    else:
        print("Usage: openlrc gui", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
