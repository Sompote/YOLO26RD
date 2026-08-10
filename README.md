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

## Pretrained weights (included)

Trained checkpoints for every scale ship in `weights/` (from-scratch, 640², fitness-selected
`best.pt`, 3-class road-damage dataset described above):

| model | file | size |
|---|---|---|
| YOLO26-RD-n | `weights/yolo26n-rd.pt` | 6.7 MB |
| YOLO26-RD-s | `weights/yolo26s-rd.pt` | 25 MB |
| YOLO26-RD-m | `weights/yolo26m-rd.pt` | 64 MB |
| YOLO26-RD-l | `weights/yolo26l-rd.pt` | 73 MB |
| YOLO26-RD-x | `weights/yolo26x-rd.pt` | 164 MB |
| stock YOLO26-s baseline (recipe-matched) | `weights/yolo26s-base.pt` | 20 MB |

**All weights are stored via [Git LFS](https://git-lfs.com)** — run `git lfs install` before
cloning (or `git lfs pull` after) to fetch the real files instead of pointer stubs. These
checkpoints predict the three classes above; for your own classes, train from the YAMLs (below).
Loading any of them requires this fork (`pip install -e .`), not PyPI ultralytics:

```python
from ultralytics import YOLO
model = YOLO("weights/yolo26s-rd.pt")
model.predict("road.jpg", conf=0.25)
```

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

### 4. Validate / test / predict / export

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640            # val
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640 split=test # test (report once)
yolo predict model=runs/detect/train/weights/best.pt source=path/to/images conf=0.25
yolo export  model=runs/detect/train/weights/best.pt format=onnx imgsz=640
```

Always evaluate at the training `imgsz` (evaluating a 640-trained model at 800/960 *reduces*
accuracy on this data). Test-time augmentation (`augment=True`) is a no-op for NMS-free `end2end`
models. Exported models need no NMS post-processing.

### Recommended recipe (evidence-based)

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
