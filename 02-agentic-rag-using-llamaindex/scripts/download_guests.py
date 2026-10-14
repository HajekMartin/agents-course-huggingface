"""Refresh the local snapshot from the public Hugging Face course dataset."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

DATASET = "agents-course/unit3-invitees"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "guests.json"


def download():
    rows = []
    total = None
    while total is None or len(rows) < total:
        query = urlencode({"dataset": DATASET, "config": "default", "split": "train",
                           "offset": len(rows), "length": 100})
        with urlopen(f"https://datasets-server.huggingface.co/rows?{query}", timeout=30) as response:
            page = json.load(response)
        if page.get("partial"):
            raise RuntimeError("Dataset server returned partial data; refusing an incomplete snapshot.")
        total = page["num_rows_total"]
        batch = [entry["row"] for entry in page["rows"]]
        if not batch and len(rows) < total:
            raise RuntimeError("Dataset pagination ended before all rows were fetched.")
        rows.extend(batch)
    if not rows or len(rows) != total:
        raise RuntimeError("Dataset snapshot has an unexpected number of rows.")
    return {"dataset": DATASET, "source_url": f"https://huggingface.co/datasets/{DATASET}",
            "downloaded_on": datetime.now(timezone.utc).date().isoformat(), "guests": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    snapshot = download()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Saved {len(snapshot['guests'])} guests to {args.output}")


if __name__ == "__main__":
    main()
