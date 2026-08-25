#!/usr/bin/env python3
"""Small deterministic API smoke test for optimization A/B comparisons."""

import argparse
import hashlib
import json

import requests


TASKS = (
    ("mul", "Compute 137 * 29. Return only the integer.", "3973"),
    ("subtract", "Compute 100000 - 45678. Return only the integer.", "54322"),
    ("sequence", "Continue: 2, 6, 12, 20, 30, __. Return only the next integer.", "42"),
    ("hex", "Convert decimal 255 to hexadecimal. Return only the two hexadecimal digits in uppercase.", "FF"),
    ("logic", "All flurps are zargs. No zargs are blue. Can any flurp be blue? Return only yes or no.", "no"),
    ("zh_mul", "计算 1234 × 7，只返回整数。", "8638"),
    ("reverse", "Reverse the ASCII string abcDEF123. Return only the reversed string.", "321FEDcba"),
    ("sort", "Sort these integers ascending: 9, -2, 5, 0. Return only a comma-separated list.", "-2,0,5,9"),
)


def normalized(text: str) -> str:
    return "".join(text.strip().lower().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000/v1/chat/completions")
    parser.add_argument(
        "--api-key", default="change-me-before-exposing-the-port"
    )
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    passed = 0
    hashes = []
    for name, prompt, expected in TASKS:
        response = requests.post(
            args.url,
            headers={"Authorization": f"Bearer {args.api_key}"},
            json={
                "model": "Qwen3.8-27B",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        digest = hashlib.sha256(text.encode()).hexdigest()
        ok = normalized(text) == normalized(expected)
        passed += int(ok)
        hashes.append(digest)
        print(
            json.dumps(
                {
                    "label": args.label,
                    "task": name,
                    "ok": ok,
                    "expected": expected,
                    "output": text,
                    "sha256": digest,
                },
                ensure_ascii=False,
            )
        )
    print(
        json.dumps(
            {
                "label": args.label,
                "passed": passed,
                "total": len(TASKS),
                "combined_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
