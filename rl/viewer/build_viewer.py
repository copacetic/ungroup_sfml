"""
Embed a replay JSON into the viewer template and write a standalone HTML file.

Usage: python3 rl/viewer/build_viewer.py replay.json out.html
"""

import json
import os
import sys


def build(replay_path, out_path):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "template.html")) as f:
        tpl = f.read()
    with open(replay_path) as f:
        data = json.load(f)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("__REPLAY_JSON__", payload)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
