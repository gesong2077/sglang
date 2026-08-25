#!/usr/bin/env python3
"""Measure the OpenAI chat path with exact post-template prompt lengths."""

import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request

from transformers import AutoTokenizer


def post(base_url: str, path: str, payload: dict, api_key: str):
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return urllib.request.urlopen(request, timeout=900)


def flush(base_url: str, api_key: str) -> None:
    with post(base_url, "/flush_cache", {}, api_key) as response:
        response.read()


def make_exact_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    """Find content whose rendered one-message chat prompt is target_tokens long."""
    def rendered_length(content: str) -> int:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    overhead = rendered_length("")
    if target_tokens <= overhead:
        raise ValueError(
            f"target {target_tokens} is not larger than chat-template overhead {overhead}"
        )

    # Qwen tokenizes each additional `` x`` as one token. Keep a fallback
    # correction loop so the helper remains exact if the tokenizer changes.
    words = max(1, target_tokens - overhead)
    for _ in range(8):
        content = " x" * words
        actual = rendered_length(content)
        if actual == target_tokens:
            return content, actual
        words += target_tokens - actual
        if words <= 0:
            break
    raise RuntimeError(f"could not construct an exact {target_tokens}-token chat prompt")


def run_once(
    base_url: str,
    model: str,
    api_key: str,
    content: str,
    expected_prompt_tokens: int,
    output_tokens: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }

    request_start = time.perf_counter()
    token_times = []
    usage = None
    with post(base_url, "/v1/chat/completions", payload, api_key) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                token_times.append(time.perf_counter())

    if not token_times:
        raise RuntimeError("stream returned no content/reasoning chunks")
    if usage is None:
        raise RuntimeError("stream returned no usage chunk")

    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    decode_seconds = max(token_times[-1] - token_times[0], 1e-9)
    decode_intervals = max(completion_tokens - 1, 0)
    ttft_seconds = token_times[0] - request_start
    return {
        "target_context": expected_prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "content_chunks": len(token_times),
        "ttft_seconds": ttft_seconds,
        "prefill_tps": prompt_tokens / ttft_seconds,
        "decode_seconds": decode_seconds,
        "decode_tps": decode_intervals / decode_seconds,
        "tpot_ms": decode_seconds * 1000 / max(decode_intervals, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--model-path", default="/mnt/models/Qwen3.8-27B-FP8")
    parser.add_argument(
        "--chat-template",
        default="/mnt/models/Qwen3.8-27B-FP8/chat_template.jinja",
    )
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--label", default="openai-chat")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    tokenizer.chat_template = Path(args.chat_template).read_text()
    all_results = []
    for context in args.contexts:
        content, actual = make_exact_prompt(tokenizer, context)
        context_results = []
        for repeat in range(args.repeats):
            flush(args.base_url, args.api_key)
            result = run_once(
                args.base_url,
                args.model,
                args.api_key,
                content,
                actual,
                args.output_tokens,
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
                    "median_prefill_tps": statistics.median(
                        item["prefill_tps"] for item in context_results
                    ),
                    "median_decode_tps": statistics.median(
                        item["decode_tps"] for item in context_results
                    ),
                    "median_ttft_seconds": statistics.median(
                        item["ttft_seconds"] for item in context_results
                    ),
                }
            ),
            flush=True,
        )

    print(json.dumps({"label": args.label, "results": all_results}), flush=True)


if __name__ == "__main__":
    main()
