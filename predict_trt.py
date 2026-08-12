#!/usr/bin/env python
"""Run inference with a YOLO26-RD TensorRT engine (or a plain .pt checkpoint).

    python predict_trt.py weights/best.engine image.jpg
    python predict_trt.py weights/best.engine path/to/images --conf 0.4 --save
    python predict_trt.py weights/best.pt     video.mp4

Works with .engine, .pt and .onnx. For .engine the imgsz must match what the engine
was built with (default 640).
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="YOLO26-RD inference")
    ap.add_argument("model", help=".engine / .pt / .onnx")
    ap.add_argument("source", help="image, folder, video, glob, URL or webcam index")
    ap.add_argument("--imgsz", type=int, default=640, help="must match the engine build size")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--device", default=0)
    ap.add_argument("--save", action="store_true", help="write annotated images/video")
    ap.add_argument("--save-txt", action="store_true", help="write YOLO-format .txt labels")
    ap.add_argument("--show-n", type=int, default=10, help="how many results to print")
    args = ap.parse_args()

    from ultralytics import YOLO

    # task="detect" is REQUIRED for .engine — a serialized engine carries no task metadata
    model = YOLO(args.model, task="detect")

    results = model.predict(
        args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
        save_txt=args.save_txt,
        verbose=False,
        stream=False,
    )

    print(f"model  : {Path(args.model).name}")
    print(f"source : {args.source}")
    print(f"images : {len(results)}\n")

    for r in results[: args.show_n]:
        print(f"{Path(r.path).name}: {len(r.boxes)} detection(s)")
        for b in r.boxes:
            x1, y1, x2, y2 = (round(v) for v in b.xyxy[0].tolist())
            print(f"    {r.names[int(b.cls)]:<16} conf={float(b.conf):.2f}  "
                  f"box=({x1},{y1})-({x2},{y2})")
    if len(results) > args.show_n:
        print(f"... {len(results) - args.show_n} more")

    total = sum(len(r.boxes) for r in results)
    s = results[-1].speed
    print(f"\ntotal detections : {total}")
    print(f"speed (last)     : {s['preprocess']:.1f} pre + {s['inference']:.1f} infer"
          f" + {s['postprocess']:.1f} post ms")
    if args.save or args.save_txt:
        print(f"saved to         : {results[0].save_dir}")


if __name__ == "__main__":
    main()
