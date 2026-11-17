"""Download the user-provided Drive snapshot (not the alternate Hugging Face split).

Requires: pip install gdown
"""
import argparse
import hashlib
import json
from pathlib import Path

import gdown

FILES = {
    "corpus.jsonl": ("1QJD0PIVAclkVHZZz55NSXrPaBxotgZCA", 115116382),
    "queries_clean.jsonl": ("1CgGplng9A6atyV-vK3MGHk9-3R9KAKxJ", 132998),
    "qrels_test.tsv": ("19-KTJ5FDewPkyE3n06QMTUbQreypYFY7", 42960),
    "noise/queries_no_diacritics.jsonl": ("15f0LYb58PuL5VX9BwujreqluTg0AIAUA", 109327),
    "noise/queries_telex_typo.jsonl": ("1Bc9E5akuRDkW-51ZrTfGTptV-NVAMC3P", 130257),
    "noise/queries_phonetic.jsonl": ("1YNuSrfVz8f8aWbBcYMSGd_6N9Pa7zVZ4", 132162),
    "noise/queries_abbrev.jsonl": ("1mFlKd3n-tgIzuIRvcer7lcl48vLgaz9V", 131197),
    "noise/queries_mixed_low.jsonl": ("1rz2BMWo1QRyEHtGjRnIUspoAIEavYw-_", 130624),
    "noise/queries_mixed_medium.jsonl": ("1xMfwHW_7ukbejvPZA9Aqr5aQvWRQ53Hv", 128153),
    "noise/queries_mixed_high.jsonl": ("17V5P0p7pM-83Qg7vcVwegl0Wl80s8Rvu", 125672),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/zalo_legal"))
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (file_id, expected_size) in FILES.items():
        output = root / name
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.is_file() or output.stat().st_size != expected_size:
            if output.exists():
                raise ValueError(f"Unexpected file size; will not overwrite: {output}")
            print(f"Downloading {name} ...", flush=True)
            result = gdown.download(id=file_id, output=str(output), quiet=False)
            if not result:
                raise RuntimeError(f"Drive download failed: {name}")
        if output.stat().st_size != expected_size:
            raise ValueError(f"Size mismatch for {name}: {output.stat().st_size}")
        manifest[name] = {"file_id": file_id, "bytes": expected_size, "sha256": sha256(output)}
        print(f"Verified {name}", flush=True)
    (root / "drive_manifest.json").write_text(
        json.dumps({"source_folder": "18fC4b6ve3zxCzdqMIWADmpmja1yP2XoD", "files": manifest},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"SUCCESS: {len(FILES)} files verified.")


if __name__ == "__main__":
    main()
