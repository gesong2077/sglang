"""Pack 64 English, 64 Chinese and 128 image-description sequences of 2048 real tokens."""

import csv
import json
import random
import argparse
import secrets
import hashlib
import shutil
import urllib.request
import tempfile
from pathlib import Path
from transformers import AutoTokenizer

from export_moss import BASE, CALIBRATION

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--seed", type=int, help="Replay a calibration seed (default: random)"
)
parser.add_argument(
    "--download", action="store_true", help="Fetch missing pinned ModelScope inputs"
)
args = parser.parse_args()
root = CALIBRATION
root.mkdir(parents=True, exist_ok=True)
for item in json.loads(Path(__file__).with_name("sources.json").read_text()):
    path = root / item["local"]
    if args.download and not path.exists():
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            fetched = Path(tmp) / item["local"]
            with (
                urllib.request.urlopen(item["url"], timeout=120) as response,
                fetched.open("wb") as f,
            ):
                shutil.copyfileobj(response, f)
            with fetched.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != item["sha256"]:
                    raise ValueError(f"Downloaded file checksum mismatch: {path.name}")
            fetched.replace(path)
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != item["sha256"]:
            raise ValueError(f"Calibration source checksum mismatch: {f.name}")
output = root / "packed.json"
if output.exists():
    raise FileExistsError(output)
seed = secrets.randbits(32) if args.seed is None else args.seed
rng = random.Random(seed)
tok = AutoTokenizer.from_pretrained(str(BASE / "tokenizer"), local_files_only=True)
packs = []
for name, count in [("alpaca-en", 64), ("alpaca-zh", 64), ("llava-detail", 128)]:
    if name.startswith("alpaca"):
        with (root / (name + ".csv")).open() as f:
            rows = [
                (
                    i,
                    "\n".join(
                        str(x[k]).strip()
                        for k in ("instruction", "input", "output")
                        if x[k]
                    ),
                )
                for i, x in enumerate(csv.DictReader(f))
            ]
    else:
        with (root / (name + ".json")).open() as f:
            rows = [
                (
                    x["id"],
                    "\n".join(
                        m["value"] for m in x["conversations"] if m["from"] == "gpt"
                    ),
                )
                for x in json.load(f)
            ]
    rng.shuffle(rows)
    ids = []
    references = []
    for key, text in rows:
        if not text.strip():
            continue
        ids.extend(tok.encode(text, add_special_tokens=False) + [tok.eos_token_id])
        references.append(key)
        if len(ids) >= 2048:
            packs.append(dict(source=name, records=references, input_ids=ids[:2048]))
            ids = []
            references = []
            count -= 1
            if count == 0:
                break
    assert count == 0
rng.shuffle(packs)
assert len(packs) == 256 and all(len(p["input_ids"]) == 2048 for p in packs)
output.write_text(json.dumps(dict(seed=seed, samples=packs), ensure_ascii=False))
print(seed, len(packs), sum(len(p["input_ids"]) for p in packs))
