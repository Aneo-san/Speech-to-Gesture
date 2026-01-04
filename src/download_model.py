"""
Simple model downloader.
Usage:
  python download_model.py --url <MODEL_URL> --out lstmp.pt

This script uses only the Python stdlib and works on Windows/macOS/Linux.
If you need to download from Hugging Face private repo, use `huggingface_hub` or provide an authenticated URL.
"""
import argparse
import sys
from urllib.request import urlopen, Request


def download(url: str, out_path: str):
    req = Request(url, headers={"User-Agent": "python-urllib/3"})
    with urlopen(req) as r, open(out_path, "wb") as f:
        chunk_size = 8192
        while True:
            chunk = r.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Direct URL to the model file")
    p.add_argument("--out", default="lstmp.pt", help="Output filename (default: lstmp.pt)")
    args = p.parse_args()

    try:
        print(f"Downloading {args.url} -> {args.out} ...")
        download(args.url, args.out)
        print("Download completed.")
    except Exception as e:
        print("Download failed:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
