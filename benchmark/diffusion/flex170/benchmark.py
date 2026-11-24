"""Run the historical Flex 170 image benchmark against the installed SGLang."""

import argparse
import base64
from collections import Counter
from contextlib import contextmanager, nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import socket
from statistics import mean
import subprocess
import sys
import threading
import time

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def launch(case, model_dir, port):
    qwen = not case.startswith("moss-")
    resident = case.endswith("-int8")
    model = read(HERE / "cases.json")[case]["model"]
    cmd = [
        str(Path(sys.executable).with_name("sglang")),
        "serve",
        "--model-path",
        str(model_dir / model),
        "--num-gpus",
        "2",
        "--tp-size",
        "2" if qwen else "1",
        "--ulysses-degree",
        "1",
        "--ring-degree",
        "1",
        "--cfg-parallel-size",
        "1" if qwen else "2",
        "--attention-backend",
        "fa" if qwen else "torch_sdpa",
        "--warmup-mode",
        "off",
        "--component-residency",
        "all=resident",
    ]
    if not resident:
        cmd.extend(
            [
                "text_encoder=layerwise-offload",
                "--layerwise-prefetch-size",
                "text_encoder=1",
            ]
        )
    if qwen:
        cmd.extend(
            [
                "--vae-config.parallel-decode-mode",
                "spatial_shard",
                "--performance-mode",
                "speed",
                "--component-attention-backends",
                "transformer=fa",
            ]
        )
    else:
        cmd.append("--disable-conditioning-cache")
    cmd.extend(["--host", "127.0.0.1", "--port", str(port)])
    env = dict(
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="4",
        CCL_SYCL_COPY_ENGINE="1",
        TOKENIZERS_PARALLELISM="false",
        LD_LIBRARY_PATH=os.pathsep.join(
            filter(
                None, [str(Path(sys.prefix) / "lib"), os.environ.get("LD_LIBRARY_PATH")]
            )
        ),
    )
    if resident:
        env.update(
            SGLANG_DIFFUSION_FLEX_CONVROT_TOKEN_PARALLEL="1",
            PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.8",
        )
    elif qwen:
        env["SGLANG_DIFFUSION_FLEX_BF16_PERSISTENT"] = "1"
    if qwen:
        for key in (
            "SGLANG_DIFFUSION_FLEX_TP2_PEER",
            "SGLANG_DIFFUSION_FLEX_TP2_PACKED_GATED",
        ):
            if key in os.environ:
                env[key] = os.environ[key]
    return cmd, env


@contextmanager
def server(cmd, env, directory, url, port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    with (directory / "server.log").open("w") as log:
        p = subprocess.Popen(
            cmd,
            env=env,
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(900):
                if p.poll() is not None:
                    raise RuntimeError(f"Server exited {p.returncode}. See {log.name}")
                try:
                    if requests.get(url + "/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"Server startup timed out. See {log.name}")
            yield
        finally:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()


def request(directory, url, payload, label, steps, paths=()):
    body = dict(payload, num_inference_steps=steps)
    if label != "priming":
        body["perf_dump_path"] = str(directory / f"{label}.json")
    with thermal_samples(paths) if paths else nullcontext() as flags:
        start = time.perf_counter()
        response = requests.post(
            url + "/v1/images/generations", json=body, timeout=2400
        )
        end = time.perf_counter()
    seconds = end - start
    response.raise_for_status()
    image = base64.b64decode(response.json()["data"][0]["b64_json"], validate=True)
    with Image.open(io.BytesIO(image)) as im:
        assert im.size == tuple(map(int, payload["size"].split("x")))
        digest = hashlib.sha256(im.convert("RGBA").tobytes()).hexdigest()
    (directory / f"{label}.png").write_bytes(image)
    row = dict(label=label, http_s=seconds, rgba_sha256=digest)
    if paths:
        sampled = [values for stamp, values in flags if start <= stamp <= end]
        row["thermal_samples"] = len(sampled)
        row["thermally_clear"] = len(sampled) >= seconds * 2 and all(
            not any(x) for x in sampled
        )
    if label != "priming":
        row["steps"] = len(read(directory / f"{label}.json")["denoise_steps_ms"])
        row["rank"] = [read(directory / f"{label}.json.rank{i}.json") for i in range(2)]
    return row


def thermal_flags():
    root = Path("/sys/class/drm")
    for card in sorted(root.glob("card[0-9]*")):
        device = card / "device/device"
        if device.is_file() and device.read_text().strip().lower() == "0x56c0":
            yield card / "gt/gt0/throttle_reason_thermal"


@contextmanager
def thermal_samples(paths):
    rows = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            rows.append(
                (time.perf_counter(), [int(path.read_text()) for path in paths])
            )
            stop.wait(0.25)

    t = threading.Thread(target=sample)
    t.start()
    try:
        yield rows
    finally:
        stop.set()
        t.join()


def run(args):
    cfg = read(HERE / "cases.json")[args.case]
    cmd, env = launch(args.case, args.models_dir.resolve(), args.port)
    if args.print_server:
        print(shlex.join(["env", *(f"{k}={v}" for k, v in env.items()), *cmd]))
        return
    if args.out is None:
        raise ValueError("--out must name a new output directory")
    directory = args.out.resolve()
    qwen = not args.case.startswith("moss-")
    if args.capture_attention and args.case not in {"base-bf16", "moss-original"}:
        raise ValueError("Capture attention with base-bf16 or moss-original")
    directory.mkdir(parents=True, exist_ok=False)
    payload = read(HERE / "requests.json")["qwen" if qwen else "moss"]
    save(
        directory / "config.json",
        dict(command=cmd, environment=env, request=payload, case=args.case),
    )
    env = {
        **os.environ,
        **env,
        "FLEX_BENCH_FAMILY": "qwen" if qwen else "moss",
        "QWEN_RESULT": str(directory),
        "PYTHONPATH": os.pathsep.join(
            filter(
                None,
                [
                    str(HERE / "instrument"),
                    str(HERE / "attention"),
                    os.getenv("PYTHONPATH"),
                ],
            )
        ),
    }
    if args.capture_attention:
        env["FLEX_CAPTURE_ATTENTION"] = "1"
    if args.case.endswith("-int8"):
        env["FLEX_CHECK_RESIDENCY"] = "1"
    paths = list(thermal_flags()) if not qwen else []
    if not qwen and (len(paths) != 2 or not all(p.is_file() for p in paths)):
        raise RuntimeError(
            "MOSS controlled timings require the two i915 Flex thermal flags"
        )
    steps = cfg["steps"]
    url = f"http://127.0.0.1:{args.port}"
    with server(cmd, env, directory, url, args.port):
        prime = request(directory, url, payload, "priming", 2 if qwen else 3)
        if qwen:
            short = request(directory, url, payload, "short", 2)
        else:
            time.sleep(180)
        hot = []
        for index in range(cfg["hot_requests"]):
            row = request(directory, url, payload, f"hot-{index+1}", steps, paths)
            hot.append(row)
        diagnostic = request(directory, url, payload, "diagnostic", steps)
    checked = hot + [diagnostic]
    valid = all(
        r["rgba_sha256"] == cfg["rgba_sha256"] and r["steps"] == steps for r in checked
    )
    if qwen:
        valid &= (
            prime["rgba_sha256"] == short["rgba_sha256"] == cfg["priming_rgba_sha256"]
        )
    else:
        valid &= all(r["thermally_clear"] for r in hot)
    ranks = []
    for rank, record in enumerate(diagnostic["rank"]):
        ops = Counter()
        for _, dtype, m, k, n, repeats in record["ops"]:
            ops[dtype] += 2 * m * k * n * repeats
        seconds = mean(
            r["rank"][rank]["seconds"] if qwen else max(v["seconds"] for v in r["rank"])
            for r in hot
        )
        dtype = (
            "torch.int8"
            if args.case.endswith("-int8") or args.case == "moss-both"
            else "torch.bfloat16"
        )
        if qwen:
            valid &= ops[dtype] == cfg["rank_operations"][rank]
        ranks.append(
            dict(
                rank=rank,
                operations=ops,
                mean_dit_s=seconds,
                dtype=dtype,
                effective=ops[dtype] / 1e12 / seconds,
            )
        )
    summary = dict(
        case=args.case,
        priming=prime,
        hot=hot,
        diagnostic=diagnostic,
        rank=ranks,
        hot_http_s=mean(r["http_s"] for r in hot),
        passes_reference=valid,
    )
    if qwen:
        summary["short"] = short
    if args.case.endswith("-int8"):
        from instrument.summarize_monitor import summarize

        residency = summarize(directory)
        save(directory / "resident-monitor-summary.json", residency)
        for info in residency["ranks"]:
            ok = info.get("strict_all_three_xpu_in_full_samples", False) and (
                info["cases"]["priming"]["actual_calls"].get("encoder.forward") == 1
                and all(
                    row["other_device_anchor_count"] == 0
                    for row in info["cases"].values()
                )
                and all(
                    c["offload_managers"] == 0
                    for sample in info["full"].values()
                    for c in sample["components"].values()
                )
            )
            ranks[info["rank"]]["resident"] = ok
            valid &= ok
        summary["passes_reference"] = valid
    save(directory / "summary.json", summary)
    print(directory / "summary.json", ranks)
    if not valid:
        raise SystemExit("Reference check failed. See summary.json for this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=read(HERE / "cases.json"))
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.getenv("FLEX_MODELS_DIR", Path.home() / "models")),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--port", type=int, default=30774)
    parser.add_argument("--print-server", action="store_true")
    parser.add_argument("--capture-attention", action="store_true")
    run(parser.parse_args())
