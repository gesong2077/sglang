#!/usr/bin/env python3
"""Measure single-request TTFT and steady decode throughput across contexts."""

import argparse
import json
import statistics
import time
import urllib.request


def post(base_url: str, path: str, payload: dict, api_key: str | None = None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    return urllib.request.urlopen(request, timeout=900)


def flush(base_url: str, api_key: str | None = None) -> None:
    with post(base_url, "/flush_cache", {}, api_key) as response:
        response.read()


def run_once(
    base_url: str, context: int, output_tokens: int, api_key: str | None = None
) -> dict:
    payload = {
        "input_ids": [100] * context,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    start = time.perf_counter()
    token_times = {}
    with post(base_url, "/generate", payload, api_key) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            obj = json.loads(data)
            count = obj.get("meta_info", {}).get("completion_tokens")
            if count is not None and count not in token_times:
                token_times[count] = time.perf_counter()

    counts = sorted(token_times)
    if len(counts) < 2:
        raise RuntimeError(f"received only {len(counts)} unique token events")
    first, last = counts[0], counts[-1]
    decode_seconds = token_times[last] - token_times[first]
    decode_tokens = last - first
    return {
        "context": context,
        "events": [first, last],
        "ttft_seconds": token_times[first] - start,
        "prefill_tps": context / (token_times[first] - start),
        "decode_seconds": decode_seconds,
        "decode_tps": decode_tokens / decode_seconds,
        "tpot_ms": decode_seconds * 1000 / decode_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--api-key")
    args = parser.parse_args()

    all_results = []
    for context in args.contexts:
        for _ in range(args.warmup_repeats):
            flush(args.base_url, args.api_key)
            run_once(args.base_url, context, args.output_tokens, args.api_key)
        context_results = []
        for repeat in range(args.repeats):
            flush(args.base_url, args.api_key)
            result = run_once(
                args.base_url, context, args.output_tokens, args.api_key
            )
            result["repeat"] = repeat + 1
            context_results.append(result)
            all_results.append(result)
            print(json.dumps({"label": args.label, **result}), flush=True)
        print(
            json.dumps(
                {
                    "label": args.label,
                    "context": context,
                    "median_decode_tps": statistics.median(
                        item["decode_tps"] for item in context_results
                    ),
                    "median_ttft_seconds": statistics.median(
                        item["ttft_seconds"] for item in context_results
                    ),
                    "median_prefill_tps": statistics.median(
                        item["prefill_tps"] for item in context_results
                    ),
                }
            ),
            flush=True,
        )

    print(json.dumps({"label": args.label, "results": all_results}), flush=True)


if __name__ == "__main__":
    main()
