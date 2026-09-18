"""Download NVIDIA runtime wheels from PyPI and extract their Windows DLLs into a runtime directory.

ort 2.0.0-rc.13 ships ONNX Runtime 1.28 built for CUDA 13 + cuDNN 9 (ort-sys dist.tsv), so the
Rust engine needs these DLLs next to the executable (or on PATH).

Usage: uv run --project tools python tools/fetch_runtime.py --out assets/runtime [--tensorrt]

--tensorrt adds TensorRT 10.16 (cu13) from pypi.nvidia.com: ORT 1.28's TensorRT EP links nvinfer_10.dll.
"""
import re
import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

PACKAGES = [
    "nvidia-cuda-runtime",
    "nvidia-cublas",
    "nvidia-cudnn-cu13",
    "nvidia-cufft",
    "nvidia-curand",
    "nvidia-nvjitlink",
    "nvidia-cuda-nvrtc",
]


TENSORRT_INDEX = "https://pypi.nvidia.com/tensorrt-cu13-libs/"
TENSORRT_WHEEL = "tensorrt_cu13_libs-10.16.1.11-py3-none-win_amd64.whl"


def tensorrt_url() -> str:
    with urllib.request.urlopen(TENSORRT_INDEX) as r:
        index = r.read().decode()
    href = re.search(r'href="([^"]*' + re.escape(TENSORRT_WHEEL) + r')', index).group(1)
    return href if href.startswith("http") else TENSORRT_INDEX + href


def wheel_url(package: str) -> str:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json") as r:
        data = json.load(r)
    for f in data["urls"]:
        if f["filename"].endswith("win_amd64.whl"):
            return f["url"]
    raise SystemExit(f"no win_amd64 wheel for {package}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--package", action="append", default=None, help="override package list")
    ap.add_argument("--tensorrt", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, wheel_url(p)) for p in (args.package or PACKAGES)]
    if args.tensorrt:
        jobs.append(("tensorrt-cu13-libs", tensorrt_url()))
    for package, url in jobs:
        print(f"{package}: {url.rsplit('/', 1)[-1]}")
        with urllib.request.urlopen(url) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        for name in z.namelist():
            if name.lower().endswith(".dll"):
                target = out / Path(name).name
                target.write_bytes(z.read(name))
                print(f"  {target.name}")


if __name__ == "__main__":
    main()
