# YOLO26-RD: NMS-Free Road-Damage Detection

An end-to-end (NMS-free) object detector for pavement distress — **alligator cracking, linear
cracks, and patching** — built on YOLO26 and revised by a data audit of real road-survey imagery.

YOLO26-RD keeps the high-resolution P2/4 branch as a **feature** path fused into the neck, but
**removes the P2 detection level** (Detect(P3, P4, P5)): on region-annotated survey imagery only
~1.4% of instances are COCO-small at 640², so a stride-4 detection level would spend 75% of all
anchors where almost no ground truth lives. Two lightweight modules complete the design:

- **LearnableContrast** (494 parameters) — a differentiable, per-tile analogue of CLAHE: local
  gamma/gain predicted from tile statistics, bilinearly blended, zero-initialized to identity,
  active at inference.
- **EdgeSPD** (+2 parameters over SPD-Conv) — Sobel-gated lossless space-to-depth downsampling at
  the P3 and P4 transitions.

This repository is a fork of Ultralytics 8.4.115 with both modules built in — a stock
`pip install ultralytics` will **not** load these models.

## Model zoo

One architecture, five scales. The **scale letter is read from the YAML filename**; all files below
are included ready to use.

| model | config | params | GFLOPs @640 | val mAP50 / 50-95 | test mAP50 / 50-95 |
|---|---|---|---|---|---|
| YOLO26-RD-n | `models/yolo26n-rd.yaml` | 3.13M | ~9 | 0.757 / 0.460 | 0.721 / 0.449 |
| YOLO26-RD-s | `models/yolo26s-rd.yaml` | 12.42M | ~31 | 0.787 / 0.469 | 0.737 / 0.455 |
| YOLO26-RD-m | `models/yolo26m-rd.yaml` | 31.61M | ~88 | 0.780 / 0.488 | 0.735 / 0.476 |
| YOLO26-RD-l | `models/yolo26l-rd.yaml` | 36.22M | ~101 | **0.809 / 0.497** | 0.738 / 0.475 |
| YOLO26-RD-x | `models/yolo26x-rd.yaml` | 81.36M | ~225 | 0.785 / 0.488 | **0.761 / 0.484** |

All figures are the fitness-selected `best.pt` (selection = val mAP50-95), trained **from scratch**
at 640² on a 7,618-image road-survey dataset (3 classes; 6,563 / 504 / 551 train/val/test), and
evaluated once per split. Note the val/test disagreement at the top of the range: scale l wins
validation, scale x wins the held-out test split — select scales on your own held-out data, not on
validation alone.

**Base model.** The unmodified YOLO26 family is included (this is a full Ultralytics fork), so the
stock baseline trains from the same install: `model=yolo26s.yaml` (or n/m/l/x).

## Result comparison vs. base YOLO26

Same dataset, identical from-scratch recipe (640², 120 epochs, MuSGD, mosaic 0.5), fitness-selected
checkpoints, single evaluation per split:

| model | params | train s/epoch | val mAP50 / 50-95 | test mAP50 / 50-95 |
|---|---|---|---|---|
| YOLO26-s (base, recipe-matched) | 10.01M | **21.7** | 0.773 / 0.469 | 0.709 / 0.445 |
| **YOLO26-RD-s** | 12.42M | 66.9 | **0.787** / 0.469 | **0.737 / 0.455** |

Seed replication (3 seeds each, test split): **YOLO26-RD-s 0.731 ± 0.007** vs
**YOLO26-s 0.718 ± 0.008** — YOLO26-RD ahead at every seed; the mean margin (+1.3 mAP50) is
consistent in direction but within the resolution of n = 3 seeds. Two honest caveats: the base
model trains ~3× faster, and if standard **pretrained weights** are used (`yolo26s.pt`), the
warm-started base model surpasses every from-scratch result here — YOLO26-RD is a *from-scratch*
architecture (its custom stem and downsamplers accept only ~39% of stock weights). At larger scale,
YOLO26-RD-x reaches **0.761** test mAP50, above every base-model scale we measured.

## How to use

### 1. Install

```bash
git clone <this-repo-url>
cd YOLO26-RD
pip install -e .
python -c "from ultralytics.nn.modules import EdgeSPD, LearnableContrast; print('YOLO26-RD fork OK')"
```

### 2. Dataset

Standard YOLO detection layout with a `data.yaml`:

```yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images
nc: 3
names: ['alligator crack', 'crack', 'patching']
```

Edit `nc`/`names` for your classes; labels are normalized `class cx cy w h` lines in `*/labels/`.

### 3. Train

Pick a scale and train (batch sizes keep the effective optimizer batch at 64 via gradient
accumulation — do not "normalize" them across scales):

```bash
# n / s (batch 32)
yolo detect train model=models/yolo26s-rd.yaml data=data.yaml \
     imgsz=640 epochs=120 batch=32 mosaic=0.5 close_mosaic=30 flipud=0.5 cos_lr=True

# m / l (batch 16)
yolo detect train model=models/yolo26l-rd.yaml data=data.yaml \
     imgsz=640 epochs=120 batch=16 mosaic=0.5 close_mosaic=30 flipud=0.5 cos_lr=True

# x (batch 8)
yolo detect train model=models/yolo26x-rd.yaml data=data.yaml \
     imgsz=640 epochs=120 batch=8 mosaic=0.5 close_mosaic=30 flipud=0.5 cos_lr=True

# base-model comparison run (stock YOLO26, same recipe)
yolo detect train model=yolo26s.yaml data=data.yaml \
     imgsz=640 epochs=120 batch=32 mosaic=0.5 close_mosaic=30 flipud=0.5 cos_lr=True
```

Python API — recommended for top-down road imagery, where exact 90° rotation is label-preserving
and was the largest single gain we measured (+3.7 test mAP50 on the base model):

```python
import albumentations as A            # pip install albumentations
from ultralytics import YOLO

model = YOLO("models/yolo26s-rd.yaml")
model.train(data="data.yaml", imgsz=640, epochs=120, batch=32,
            mosaic=0.5, close_mosaic=30, flipud=0.5, cos_lr=True,
            augmentations=[A.RandomRotate90(p=0.5)])
```

Notes:
- A `.yaml` model trains **from scratch**; `pretrained=True` is inert. That is the intended regime
  for YOLO26-RD (see comparison caveats above).
- Resume with `yolo detect train resume model=.../weights/last.pt` (re-pass `augmentations=`).
- `best.pt` is selected on val mAP50-95; report it, not per-epoch peaks.

### 4. Validate / test

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640            # val
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640 split=test # test (report once)
```

Always evaluate at the training `imgsz` (evaluating a 640-trained model at 800/960 *reduces*
accuracy on this data). Test-time augmentation (`augment=True`) is a no-op for NMS-free `end2end`
models. Exported models need no NMS post-processing.

### 5. Inference

```bash
# CLI
yolo predict model=weights/best.pt source=path/to/images conf=0.25

# helper script — prints per-image detections and the speed breakdown
python predict_trt.py weights/best.pt path/to/images --conf 0.25 --save
```

Python API:

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")                  # or "weights/best.engine"
results = model.predict("image.jpg", imgsz=640, conf=0.25, device=0)

for r in results:
    for b in r.boxes:
        name = r.names[int(b.cls)]               # 'alligator crack' | 'crack' | 'patching'
        conf = float(b.conf)
        x1, y1, x2, y2 = b.xyxy[0].tolist()      # pixels in the ORIGINAL image, not 640x640
        print(name, round(conf, 2), (x1, y1, x2, y2))
```

`source=` accepts an image, folder, glob, video, URL, or webcam index. Add `save=True` for
annotated output, `save_txt=True` for YOLO-format labels. For long videos or large folders use
`stream=True` to iterate lazily instead of building all results in memory.

### 6. Export to TensorRT (FP16)

TensorRT gives roughly an **8-11x** speedup over eager PyTorch on the same GPU.

```bash
pip install -r requirements-tensorrt.txt

python export_tensorrt.py weights/best.pt                     # FP16, imgsz 640, batch 1
python export_tensorrt.py weights/best.pt --imgsz 640 --batch 8
python export_tensorrt.py weights/best.pt --int8 --data data.yaml
```

Equivalent one-liners:

```python
from ultralytics import YOLO
YOLO("weights/best.pt").export(format="engine", half=True, imgsz=640, device=0)
```
```bash
yolo export model=weights/best.pt format=engine half=True imgsz=640 device=0
```

Then run it exactly like a `.pt`:

```bash
python predict_trt.py weights/best.engine path/to/images
```

Export goes `.pt -> .onnx -> .engine` and takes ~40-60 s; the intermediate `.onnx` can be deleted.

**Four things to know about engines:**

1. **`task="detect"` is required** when loading an `.engine` in the Python API — a serialized
   engine carries no task metadata, so Ultralytics cannot infer it:
   ```python
   YOLO("best.engine", task="detect")
   ```
2. **`imgsz` and `batch` are baked in at build time.** Build separate engines if you need more
   than one input size or batch size.
3. **Engines are not portable.** They are compiled for the specific GPU architecture, TensorRT
   version and CUDA version used at build time. Re-export from the `.pt` on each target machine.
4. **Discard the first inference when timing** — the first call pays a one-off warmup cost
   (roughly 25 ms vs ~2 ms settled).

#### Measured TensorRT FP16 latency

640x640, batch 1, RTX 5090, TensorRT 10.16, released YOLO26-RD weights:

| model | params | preprocess | inference | postprocess | total | FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26-RD-n | 3.00M | 2.20 | 1.50 | 0.45 | 4.16 | 241 |
| YOLO26-RD-s | 11.93M | 2.24 | 1.64 | 0.36 | 4.24 | 236 |
| YOLO26-RD-m | 30.18M | 2.12 | 1.89 | 0.44 | 4.45 | 225 |
| YOLO26-RD-l | 34.78M | 2.17 | 2.64 | 0.35 | 5.16 | 194 |
| YOLO26-RD-x | 78.17M | 2.14 | 3.31 | 0.45 | 5.90 | 170 |

*(ms; stock YOLO26-s for reference: 1.22 ms inference)*

Two practical notes. **Postprocessing is only 0.35-0.45 ms** because the `end2end` head needs no
NMS. And **preprocessing costs more than inference** for every scale up to `m` — that is resizing
full-resolution source imagery down to 640, so feeding pre-resized 640x640 images is a larger win
than switching to a smaller model.

#### Export troubleshooting

| symptom | cause and fix |
|---|---|
| `CUDA driver version is insufficient for CUDA runtime version` (error 35) | `pip install tensorrt` resolved to a CUDA 13 build. Install `tensorrt-cu12` explicitly (or `tensorrt-cu13` if your driver supports CUDA 13). |
| `AttributeError: ... has no attribute 'EXPLICIT_BATCH'` | TensorRT 11 removed that flag. Pin `tensorrt-cu12>=10.0,<11.0`. |
| `ModuleNotFoundError: No module named 'onnxscript'` | The torch>=2.6 ONNX exporter needs it: `pip install onnxscript`. |
| `WARNING: ... requires precision-lose casting` | Harmless — TensorRT noting an FP16 cast, expected with `half=True`. |
| Engine fails to load on another machine | Engines are hardware/version specific. Re-export from the `.pt` there. |

### 7. Recommended recipe (evidence-based)

- `mosaic=0.5` with a generous `close_mosaic` — full mosaic truncates near-frame-size distress
  boxes and degraded long runs in our experiments.
- `flipud=0.5` and `A.RandomRotate90(p=0.5)` — top-down imagery has no canonical orientation;
  90° rotations are label-exact for axis-aligned boxes.
- Train and validate at the same `imgsz`.
- For class imbalance: duplicate rare-class images in a `train:` `.txt` list (duplicates are kept),
  or set `cls_pw` (inverse-frequency^cls_pw class weighting; 0.0 disables).

## Module reference

Both modules are registered in `ultralytics/nn/tasks.py` and usable in any model YAML:

```yaml
- [-1, 1, LearnableContrast, [3]]   # input stem; args: [channels, hidden_width, tile_grid]
- [-1, 1, EdgeSPD, [256, 3]]        # lossless downsample /2; args: [out_channels, fusion_kernel]
```

Implementation: `ultralytics/nn/modules/conv.py` (`EdgeSPD`, `LearnableContrast`).

## Citation

```bibtex
@article{youwai2026yolo26rd,
  title   = {YOLO26-RD: A Data-Audited, Contrast-Enhanced, Edge-Guided NMS-Free Detector
             for Road-Damage Detection},
  author  = {Youwai, Sompote and Chaipetch, Pawatorn Awarotorn},
  year    = {2026},
  note    = {Preprint}
}
```

## License

AGPL-3.0, inherited from the Ultralytics codebase this fork is built on.
