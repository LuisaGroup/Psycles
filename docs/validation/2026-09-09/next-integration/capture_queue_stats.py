"""Strictly capture two retained 64-spp scheduler logs; never run a renderer.

These are diagnostic counters, not a paired performance benchmark. Preserve all
seven dispatch ranges: the final 32-spp range is only half of the rendered work.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def parse(path):
    text = path.read_text()
    batches, aliases = [], []
    for raw in text.splitlines():
        line = raw.split("[info] ", 1)[-1]
        if line.startswith("Wavefront before-resume batch alias: "):
            match = re.fullmatch(r"Wavefront before-resume batch alias: queue=(\d+) boundary=(\d+)"
                                 r" -> queue=(\d+) boundary=(\d+) continuation=(\d+)\.", line)
            require(match is not None and not batches, "invalid/late alias declaration")
            aliases.append(dict(zip(("queue", "boundary", "target_queue", "target_boundary", "continuation"),
                                    map(int, match.groups()))))
        elif line.startswith("Wavefront stats: "):
            match = re.fullmatch(r"Wavefront stats: iterations=(\d+) generated=(\d+) resumed=(\d+) "
                                 r"extensions=(\d+) gather_scan=(\d+) compact_scan=(\d+) "
                                 r"max_scan=(\d+) max_active=(\d+) elapsed_ms=([\d.]+)", line)
            require(match is not None, "invalid stats record")
            keys = ("iterations", "generated", "resumed", "extensions", "gather_scan", "compact_scan",
                    "max_scan", "max_active")
            stats = dict(zip(keys, map(int, match.groups()[:-1])))
            stats["elapsed_ms"] = float(match[9])
            batches.append(dict(stats=stats, continuations={}, extensions={}, auxiliaries={}))
        elif line.startswith("Wavefront continuation: "):
            match = re.fullmatch(r"Wavefront continuation: index=(\d+) token=(\d+) name='([^']+)' "
                                 r"dispatches=(\d+) executed=(\d+) peak_queued=(\d+)\.", line)
            require(match is not None and batches, "invalid continuation record")
            index, token, name, dispatches, executed, peak = match.groups()
            row = dict(token=int(token), name=name, dispatches=int(dispatches),
                       executed=int(executed), peak_queued=int(peak))
            group, key = "continuations", int(index)
            require(key not in batches[-1][group], "duplicate continuation")
            batches[-1][group][key] = row
        elif line.startswith("Wavefront Extension: "):
            match = re.fullmatch(r"Wavefront Extension: queue=(\d+) boundary=(\d+) extension=(\d+) "
                                 r"schema='([^']+)' handler='([^']+)' dispatches=(\d+) executed=(\d+) "
                                 r"peak_queued=(\d+)\.", line)
            require(match is not None and batches, "invalid extension record")
            queue, boundary, extension, schema, handler, dispatches, executed, peak = match.groups()
            row = dict(boundary=int(boundary), extension=int(extension), schema=schema, handler=handler,
                       dispatches=int(dispatches), executed=int(executed), peak_queued=int(peak))
            group, key = "extensions", int(queue)
            require(key not in batches[-1][group], "duplicate extension")
            batches[-1][group][key] = row
        elif line.startswith("Wavefront auxiliary: "):
            match = re.fullmatch(r"Wavefront auxiliary: name='([^']+)' dispatches=(\d+) executed=(\d+) "
                                 r"peak_queued=(\d+)\.", line)
            require(match is not None and batches, "invalid auxiliary record")
            name, dispatches, executed, peak = match.groups()
            require(name not in batches[-1]["auxiliaries"], "duplicate auxiliary")
            batches[-1]["auxiliaries"][name] = dict(dispatches=int(dispatches), executed=int(executed),
                                                    peak_queued=int(peak))
    configs = re.findall(r"path coroutine: subroutines=(\d+) frame_fields=(\d+) frame_bytes=(\d+) "
                         r"capacity=(\d+)\.", text)
    renders = re.findall(r"^Rendered (\d+)x(\d+) at (\d+) spp from absolute sample range "
                         r"\[(\d+), (\d+)\) of (\d+) in ", text, re.M)
    require(len(configs) == len(renders) == 1, "missing/ambiguous frame or terminal render record")
    require(tuple(map(int, renders[0])) == (2048, 858, 64, 0, 64, 64), "different workload")
    require(len(batches) == 7, "expected all seven dispatch ranges")
    frame = dict(zip(("stages", "fields", "bytes", "capacity"), map(int, configs[0])))
    require(frame["stages"] == 6, "different continuation set")
    samples = []
    for batch in batches:
        stats, cont, ext, aux = (batch[k] for k in ("stats", "continuations", "extensions", "auxiliaries"))
        require(set(cont) == set(range(6)) and set(ext) == set(range(6, 12)), "incomplete batch queues")
        require(set(aux) == {"shade_light_nee", "intersect_shadow", "shade_shadow"}, "incomplete auxiliaries")
        require(stats["generated"] == cont[0]["executed"], "entry/generated count differs")
        require(stats["resumed"] == sum(row["executed"] for i, row in cont.items() if i != 0),
                "resumed/continuation count differs")
        require(stats["extensions"] == sum(row["executed"] for row in ext.values()) == cont[5]["executed"],
                "extension/surface count differs")
        require(sum(row["dispatches"] for row in ext.values()) == cont[5]["dispatches"],
                "extension/surface dispatch count differs")
        require(stats["max_scan"] <= frame["capacity"] and stats["max_active"] <= frame["capacity"],
                "capacity overflow")
        spp, remainder = divmod(stats["generated"], 2048 * 858)
        require(remainder == 0, "fractional sample batch")
        samples.append(spp)
    require(samples == [1, 1, 2, 4, 8, 16, 32], "different/incomplete sample batching")

    def aggregate(group):
        result = {}
        counters = {"dispatches", "executed", "peak_queued"}
        for key, first in batches[0][group].items():
            rows = [b[group][key] for b in batches]
            metadata = {k: v for k, v in first.items() if k not in counters}
            require(all({k: v for k, v in row.items() if k not in counters} == metadata for row in rows),
                    "batch metadata drift")
            result[key] = {**metadata, "dispatches": sum(r["dispatches"] for r in rows),
                           "executed": sum(r["executed"] for r in rows),
                           "peak_queued": max(r["peak_queued"] for r in rows)}
        return result

    totals = {key: sum(b["stats"][key] for b in batches) for key in
              ("iterations", "generated", "resumed", "extensions", "gather_scan", "compact_scan")}
    require(totals["generated"] == 112459776, "whole-render primary count differs")
    return dict(log=identity(path), frame=frame, samples=samples, aliases=aliases, totals=totals,
                continuations=aggregate("continuations"), extensions=aggregate("extensions"),
                auxiliaries=aggregate("auxiliaries"), batches=batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_log", type=Path)
    parser.add_argument("fresh_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite capture")
    old, fresh = parse(args.old_log), parse(args.fresh_log)
    require(not old["aliases"], "historical log unexpectedly has batching aliases")
    expected = [dict(queue=q, boundary=b, target_queue=6, target_boundary=1, continuation=5)
                for q, b in zip(range(7, 12), (2, 10, 12, 14, 16))]
    require(fresh["aliases"] == expected, "fresh canonical alias mapping differs")
    require(old["frame"] == fresh["frame"], "frame layout summary changed")
    require({q for q, row in old["extensions"].items() if row["executed"]} == {7, 8}, "old active queues differ")
    require({q for q, row in fresh["extensions"].items() if row["executed"]} == {6}, "fresh active queue differs")
    result = dict(schema="psycles.next-integration.queue-diagnostic.v1", capture_source=identity(Path(__file__)),
                  scope="64-spp diagnostic counters; not a paired performance result or proof of path identity",
                  old=old, fresh=fresh, total_deltas={k: fresh["totals"][k] - v for k, v in old["totals"].items()})
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, run in (("old", old), ("fresh", fresh)):
        print(name, run["totals"], "surface", run["continuations"][5])
    print("Saved", args.output)


if __name__ == "__main__":
    main()
