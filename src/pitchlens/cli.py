"""``pitchlens`` command-line entry point.

    pitchlens generate --seed 6 --narrative late_comeback --out data/synthetic/demo
    pitchlens serve --port 8000
    pitchlens schema            # export JSON Schemas to docs/schema
    pitchlens footage --start 3721 --duration 30
    pitchlens vision --video clip.mp4 --calib calib.json --out outputs/vision
"""

from __future__ import annotations

import sys


def main() -> None:
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("--help", [])
    sys.argv = [f"pitchlens {cmd}", *rest]
    if cmd == "generate":
        from pitchlens.synth.generate import main as run
    elif cmd == "footage":
        from pitchlens.footage.broadcast import main as run
    elif cmd == "vision":
        from pitchlens.vision.pipeline import main as run
    elif cmd == "calibrate":
        from pitchlens.vision.calibrate import main as run
    elif cmd == "schema":
        from pathlib import Path

        from pitchlens.schema.export import export

        def run() -> None:
            for p in export(Path("docs/schema")):
                print(p)
    elif cmd == "serve":
        import argparse

        import uvicorn

        def run() -> None:
            ap = argparse.ArgumentParser()
            ap.add_argument("--port", type=int, default=8000)
            ap.add_argument("--host", default="127.0.0.1")
            a = ap.parse_args()
            uvicorn.run("pitchlens.api.app:app", host=a.host, port=a.port)
    else:
        print(__doc__)
        return
    run()


if __name__ == "__main__":
    main()
