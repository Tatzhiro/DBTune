#!/usr/bin/env python3
"""Derive the online-only knob file: keep knobs MySQL can change with SET GLOBAL (no restart).

    python scripts/make_online_knob_file.py experiment/gen_knobs/mysql_perf_8.0.json \
        experiment/gen_knobs/mysql_perf_8.0_online.json
"""
import json
import sys


def is_dynamic(spec: dict) -> bool:
    return str(spec.get("dynamic", "")).lower() in ("yes", "true", "1")


def main(src: str, dst: str) -> None:
    knobs = json.load(open(src))
    online = {name: spec for name, spec in knobs.items() if is_dynamic(spec)}
    with open(dst, "w") as f:
        json.dump(online, f, indent=4)
    dropped = sorted(set(knobs) - set(online))
    print("%d knobs -> %d online; dropped %d static: %s" % (len(knobs), len(online), len(dropped), ", ".join(dropped)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
