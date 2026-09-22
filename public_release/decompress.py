import gzip
import shutil
from pathlib import Path

for gz in sorted(Path(__file__).resolve().parent.rglob("*.gz")):
    out = gz.with_suffix("")
    if not out.exists():
        with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        print(out)
