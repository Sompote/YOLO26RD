#!/usr/bin/env python
"""Convert a YOLO26-RD .pt checkpoint to a TensorRT FP16 engine.

    python export_tensorrt.py weights/best.pt
    python export_tensorrt.py weights/best.pt --imgsz 640 --batch 8
    python export_tensorrt.py weights/best.pt --int8 --data data.yaml

The .engine is written next to the .pt. Build takes roughly 40-60 s.

Requirements (see requirements-tensorrt.txt) — the version pins matter:
    pip install onnx onnxslim onnxscript "tensorrt-cu12>=10.0,<11.0"
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Export YOLO26-RD to TensorRT FP16")
    ap.add_argument("weights", help="path to .pt checkpoint")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size, baked into the engine")
    ap.add_argument("--batch", type=int, default=1, help="batch size, baked into the engine")
    ap.add_argument("--device", default=0, help="CUDA device; the engine is compiled for THIS GPU")
    ap.add_argument("--int8", action="store_true", help="INT8 instead of FP16 (needs --data)")
    ap.add_argument("--data", default=None, help="data.yaml, required for INT8 calibration")
    ap.add_argument("--no-verify", action="store_true", help="skip the load-back check")
    args = ap.parse_args()

    if args.int8 and not args.data:
        sys.exit("--int8 requires --data <data.yaml> for calibration")

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        sys.exit("TensorRT not installed. Run:\n"
                 '  pip install onnx onnxslim onnxscript "tensorrt-cu12>=10.0,<11.0"')

    from ultralytics import YOLO

    pt = Path(args.weights)
    if not pt.exists():
        sys.exit(f"not found: {pt}")

    print(f"converting : {pt}")
    print(f"imgsz={args.imgsz}  batch={args.batch}  "
          f"precision={'INT8' if args.int8 else 'FP16'}\n")

    kw = dict(format="engine", imgsz=args.imgsz, batch=args.batch, device=args.device)
    if args.int8:
        kw.update(int8=True, data=args.data)
    else:
        kw.update(half=True)

    t0 = time.perf_counter()
    out = YOLO(str(pt)).export(**kw)
    dt = time.perf_counter() - t0

    eng = Path(out)
    print(f"\nengine : {eng}")
    print(f"size   : {eng.stat().st_size / 1e6:.1f} MB")
    print(f"build  : {dt:.0f} s")

    if not args.no_verify:
        m = YOLO(str(eng), task="detect")          # task= is REQUIRED for .engine
        print(f"verify : loaded OK ({len(m.names)} classes)")

    print(f"\nrun it with:\n  python predict_trt.py {eng} <image-or-folder>")


if __name__ == "__main__":
    main()
