# RetailAudit-Net
### Automated Retail Shelf Auditing and Planogram Compliance System
**Group 17** | Sahil Chordia (121133106) | Harshwardhan Rathore (121281982)

---

## Overview

RetailAudit-Net is a computer vision pipeline that automates retail shelf auditing. It detects products on dense retail shelves, identifies out-of-stock gaps, segments shelf rows, encodes product features using CLIP, and scores planogram compliance — all from a single shelf image.

---

## Project Structure

```
cv_final_project1/
├── src/
│   ├── download_dataset.py     dataset downloader (Kaggle API)
│   ├── train.py                Phase 1: YOLOv11 training on SKU-110K
│   ├── detect.py               Phase 1: product detection on test images
│   ├── shelf_structure.py      Phase 2: shelf row segmentation and gap detection
│   ├── feature_encoding.py     Phase 3: CLIP feature encoding and clustering
│   ├── planogram.py            Phase 4: planogram compliance scoring
│   ├── evaluate.py             full evaluation with metrics and plots
│   └── pipeline.py             end-to-end pipeline (all 4 phases)
├── data/
│   ├── raw/SKU110K_fixed/
│   │   ├── images/train|val|test
│   │   └── labels/train|val|test
│   └── sku110k.yaml
├── models/
│   └── weights/
├── results/
│   ├── retailaudit_yolo11/weights/best.pt
│   ├── detections/
│   ├── planogram/
│   ├── pipeline_output/
│   ├── evaluation_results.json
│   └── evaluation_plot.png
└── retail_audit_env/
```

---

## Environment Setup

**Requirements:** Python 3.12+, macOS with Apple Silicon (MPS) or CUDA GPU

```bash

cd ~/Documents/computer_vision_jwu
mkdir cv_final_project1 && cd cv_final_project1


python3 -m venv retail_audit_env
source retail_audit_env/bin/activate


pip install --upgrade pip
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
pip install numpy matplotlib pillow scikit-learn scipy tqdm
pip install git+https://github.com/openai/CLIP.git
```

**Verify installation:**
```bash
python3 -c "
import torch, cv2, ultralytics, clip
print('PyTorch:', torch.__version__)
print('MPS Available:', torch.backends.mps.is_available())
print('OpenCV:', cv2.__version__)
print('Ultralytics:', ultralytics.__version__)
"
```

---

## Dataset

**SKU-110K** — Dense retail shelf dataset

| Split | Images | Avg Objects/Image |
|-------|--------|-------------------|
| Train | 8,185  | ~147              |
| Val   | 584    | ~147              |
| Test  | 2,920  | ~147              |

Download from: https://www.kaggle.com/datasets/thedatasith/sku110k-annotations

Place extracted folder at: `data/raw/SKU110K_fixed/`

Labels are pre-formatted in YOLO format (class cx cy w h normalized).

---

## Pipeline Architecture

### Phase 1 — Dense Product Detection (YOLOv11)

Trains YOLOv11n on SKU-110K to detect products on dense retail shelves.

```bash
python3 src/train.py
python3 src/detect.py
```

- Model: YOLOv11n (2.59M parameters, 6.4 GFLOPs)
- Backbone: CSPDarknet + PANet
- Loss: CIoU box regression + Distribution Focal Loss
- Device: Apple MPS (M4) / CUDA

### Phase 2 — Shelf Structure and Perspective Normalization (OpenCV)

Segments detected products into shelf rows and detects out-of-stock gaps using geometric analysis.

```bash
python3 src/shelf_structure.py
```

- Row detection via Y-centroid clustering
- Horizontal line detection via Hough Transform
- Gap detection using normalized spacing thresholds

### Phase 3 — Feature Encoding (CLIP ViT-B/32)

Encodes cropped product regions into high-dimensional embeddings using OpenAI CLIP for visual similarity analysis.

```bash
python3 src/feature_encoding.py
```

- Model: CLIP ViT-B/32 (pretrained on contrastive objective)
- Produces 512-dim normalized embeddings per product
- Clusters visually similar products via cosine similarity

### Phase 4 — Spatial Planogram Alignment (Graph Matching)

Compares real detected shelf state against an ideal planogram layout using spatial constraint matching.

```bash
python3 src/planogram.py
```

- Builds ideal planogram from uniform spacing assumption
- Matches detected positions to ideal positions
- Outputs compliance score and violation map

---

## Running the Full Pipeline

Run all 4 phases on a single image:

```bash

python3 src/pipeline.py


python3 src/pipeline.py data/raw/SKU110K_fixed/images/test/test_1270.jpg
```

Output is saved to `results/pipeline_output/`.

---

## Evaluation

Run full quantitative evaluation on 20 test images:

```bash
python3 src/evaluate.py
```

Outputs:
- Per-image metrics printed to terminal
- `results/evaluation_results.json` — full metrics in JSON
- `results/evaluation_plot.png` — 4-panel evaluation chart

---

## Results

### Training (10 epochs, 20% dataset, img size 416)

| Epoch | box_loss | mAP50 | mAP50-95 |
|-------|----------|-------|----------|
| 1     | 2.138    | 0.282 | 0.096    |
| 5     | 1.734    | 0.536 | 0.257    |
| 10    | 1.637    | 0.616 | 0.322    |
| Best  | —        | 0.641 | 0.334    |

### Evaluation (20 test images)

| Metric                  | Score  |
|-------------------------|--------|
| Avg Precision           | 0.791  |
| Avg Recall              | 0.674  |
| Avg F1 Score            | 0.722  |
| Avg Planogram Compliance| 60.6%  |
| Total OOS Gaps Detected | 83     |

### Sample Pipeline Output (test_588.jpg)

| Phase         | Result                        |
|---------------|-------------------------------|
| Detection     | 183 products detected         |
| Shelf Structure| 9 rows, 1 OOS gap            |
| CLIP Encoding | 30 embeddings generated       |
| Planogram     | 79% compliance, 37 violations |

---

## Evaluation Metrics

- **Precision / Recall / F1** — computed against ground truth YOLO labels at IoU 0.5
- **mAP50 / mAP50-95** — mean average precision from Ultralytics validation
- **Planogram Compliance Index** — fraction of products within 8% normalized distance of ideal position
- **OOS Recall** — number of empty shelf gaps detected per image

---

## Key Design Choices

| Choice | Reason |
|--------|--------|
| YOLOv11n over R-CNN | Faster inference, better dense detection |
| Oriented bounding boxes not used (standard mode) | SKU-110K labels are axis-aligned |
| CLIP ViT-B/32 | Zero-shot visual similarity without fine-tuning |
| Row clustering via Y-centroid | Simple, effective, no extra annotation needed |
| Ideal planogram from uniform spacing | No reference planogram available in SKU-110K |

---

## Fallback Plan

If full pipeline fails, the system falls back to binary detection only:

- Class A: Product (In-Stock)
- Class B: Empty Shelf (Out-of-Stock)

Same YOLOv11 model, simplified 2-class output for basic OOS monitoring.

---

## References

1. Real-time Planogram Compliance Application using Computer Vision and Virtual Shelves — Scientific Reports, 2025
2. Context-Aware Fine-Grained Product Recognition on Grocery Shelves — IEEE Access, 2025
3. Precise Detection in Densely Packed Scenes — CVPR (EM-merger)
4. SKU-110K Dataset — Goldman et al., CVPR 2019
5. CLIP: Learning Transferable Visual Models from Natural Language Supervision — Radford et al., 2021

---

## Originality

This project is an entirely original implementation for this course. No code was reused from prior coursework or external project repositories. YOLOv11, OpenCV, and CLIP are used as pretrained/library components integrated into a novel retail auditing pipeline.