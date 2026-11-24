"""Read partial or completed all-resident probes from a benchmark's result directory."""

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(folder):
    folder = Path(folder)
    result = {"directory": str(folder), "ranks": []}
    for rank in (0, 1):
        path = folder / f"residency.rank{rank}.jsonl"
        if not path.exists():
            result["ranks"].append({"rank": rank, "snapshots": 0})
            continue
        with path.open() as stream:
            data = [json.loads(line) for line in stream if line.endswith("\n")]
        samples = [s for s in data if "xpu" in s]
        rich = [s for s in samples if "components" in s]
        by_case = {}
        wrong = []
        missing = []
        for row in rich:
            where = f'{row["case"]}/{row["point"]}/{row["step"]}'
            if set(row["components"]) != {"text_encoder", "transformer", "vae"}:
                missing.append(where)
            for role, c in row["components"].items():
                off = {}
                for key, size in c["logical_bytes"].items():
                    kind, dev, dtype = key.split("|")
                    if dev != f"xpu:{rank}" and size:
                        off[key] = size
                if off:
                    wrong.append(dict(at=where, component=role, elsewhere_bytes=off))
        for case in dict.fromkeys(s["case"] for s in data):
            selected = [s for s in samples if s["case"] == case]
            if not selected:
                continue
            calls = Counter()
            for row in data:
                if row["case"] == case and row.get("point") == "calls":
                    calls.update(row["calls"])
            anchor_mismatches = []
            for s in selected:
                for role, anchors in s.get("anchors", {}).items():
                    for key, (dev, nbytes) in anchors.items():
                        if nbytes and dev != f"xpu:{rank}":
                            anchor_mismatches.append(
                                f'{s["point"]}/{s["step"]}/{role}/{key}/{dev}'
                            )
            by_case[case] = dict(
                snapshots=len(selected),
                actual_calls=dict(calls),
                min_sampled_driver_free=min(s["xpu"]["driver_free"] for s in selected),
                lifetime_peak_allocator_allocated=max(
                    s["xpu"]["allocator_peak_allocated"] for s in selected
                ),
                lifetime_peak_allocator_reserved=max(
                    s["xpu"]["allocator_peak_reserved"] for s in selected
                ),
                cpu_monitor_ms=sum(s["monitor_cpu_ms"] for s in selected),
                other_device_anchor_count=len(anchor_mismatches),
                other_device_anchor_examples=anchor_mismatches[:8],
            )
        # One complete census at each execution boundary, including actual storage
        # sizes of aliased checkpoints and non-Parameter peer-projection layouts.
        witnesses = {}
        for row in rich:
            key = f'{row["case"]}/{row["point"]}/{row["step"]}'
            witnesses[key] = dict(
                manifest=row["manifest"],
                components={
                    role: {
                        k: c[k]
                        for k in (
                            "logical_bytes",
                            "unique_storage_bytes",
                            "exclusive_storage_bytes",
                            "embeddings",
                            "offload_managers",
                            "tp_layouts",
                        )
                    }
                    for role, c in row["components"].items()
                },
            )
        result["ranks"].append(
            dict(
                rank=rank,
                snapshots=len(samples),
                cases=by_case,
                strict_all_three_xpu_in_full_samples=bool(rich)
                and not missing
                and not wrong,
                missing_components=missing,
                other_device_censuses=wrong,
                full=witnesses,
            )
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    summary = summarize(args.directory)
    output = args.out or args.directory / "resident-monitor-summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    for rank in summary["ranks"]:
        print(
            rank["rank"],
            rank["snapshots"],
            rank.get("strict_all_three_xpu_in_full_samples"),
            rank.get("cases", {}),
        )
