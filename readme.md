# RetailAudit-Net

Automated Retail Shelf Auditing and Planogram Compliance System
Group 17 - Harshwardhan Rathore (121281982), Sahil Chordia (121133106)

## What is this

So basically this is our final project for the computer vision class. We built a pipeline that takes a picture of a retail shelf and tells you whats on it, whats missing, and whether the products are placed correctly. It uses YOLOv11 for detecting all the products, OpenCV for figuring out shelf rows and gaps, CLIP for product features, and some math to score how well the shelf matches an ideal layout.

## Live Demo on Hugging Face

We deployed it so the anyone can just open a browser and try it without installing anything.

Link: https://huggingface.co/spaces/harsh432/retailaudit-net

How to test it:
1. Open the link
2. Upload any retail shelf image or drag and drop
3. Click Run Audit
4. First run takes around 30 seconds because the server downloads CLIP weights the first time. After that its like 5 to 10 seconds per image.
5. You will see the annotated image on the right with all the product boxes and a text summary saying how many products were detected, how many rows, gaps etc.

If the Space looks asleep just wait a bit, free Spaces auto sleep after 48 hours of no use but they wake up on their own when someone opens the URL.

### Deployment

| Used | value |
|-------|-------|
| platform | Hugging Face Spaces |
| SDK | Gradio 5.49.0 |
| Python | 3.11 |
| Hardware | CPU Basic |
| Weights | best.pt from our training, 5.4 MB |

The Space has 4 files:
- `app.py` - the Gradio interface that wraps the whole pipeline
- `best.pt` - our trained YOLO weights
- `requirements.txt` - server dependencies
- `README.md` - the Space card

requirements.txt looks like this:
```
torch
torchvision
ultralytics
opencv-python-headless
numpy
Pillow
gradio
git+https://github.com/openai/CLIP.git
ftfy
regex
```

We had to use `opencv-python-headless` instead of regular `opencv-python` because HF servers don't have a display, regular opencv kept failing on import.

### How we deployed it

If anyone wants to redo this project:
```bash
hf auth login
git clone https://huggingface.co/spaces/<username>/<spacename>
cd <spacename>
cp <project>/results/retailaudit_yolo11/weights/best.pt .

git lfs install
git lfs track "*.pt"
git add .gitattributes
git add .
git commit -m "first commit"
git push
```


## Folder Structure

```
cv_final_project1/                   
├── src/
│   ├── train.py              
│   ├── detect.py             
│   ├── shelf_structure.py    
│   ├── feature_encoding.py   
│   ├── planogram.py          
│   ├── evaluate.py           
│   └── pipeline.py           
├── data/
│   └── raw/SKU110K_fixed/    
├── results/
│   ├── retailaudit_yolo11/weights/best.pt
│   ├── detections/
│   ├── planogram/
│   ├── pipeline_output/
│   ├── evaluation_results.json
│   └── evaluation_plot.png
└── retail_audit_env/         
```

## Setup

We used Python 3.12 on a Mac with Apple Silicon (M4) but it should work on CUDA too.

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

To check everything installed:
```bash
python3 -c "import torch, cv2, ultralytics, clip; print(torch.__version__, cv2.__version__, ultralytics.__version__)"
```

If you also want to push to HF:
```bash
pip install huggingface_hub gradio
```

## Dataset

We used SKU-110K which is basically the standard dataset for dense retail shelf detection. Around 11k images with about 147 products per image on average.

Download from: https://www.kaggle.com/datasets/thedatasith/sku110k-annotations

Then put it in `data/raw/SKU110K_fixed/`. Labels are already in YOLO format so no conversion needed.

| Split | Images |
|-------|--------|
| Train | 8,185 |
| Val | 584 |
| Test | 2,920 |

## How the Pipeline Works

There are 4 phases. You can run each one separately or use `pipeline.py` to run all of them on one image.

### Phase 1 - YOLOv11 detection
This trains YOLOv11n on SKU-110K and detects products on a shelf.

```bash
python3 src/train.py
python3 src/detect.py
```

We used YOLOv11n the nano version because it is small and fast. Backbone is CSPDarknet+PANet, loss is CIoU + Distribution Focal Loss. We trained on Apple MPS.

### Phase 2 - Shelf structure (OpenCV)
After we have all the boxes, we group them into rows by clustering their Y-centroids and then look for gaps that are bigger than expected, those are the out-of-stock spots.

```bash
python3 src/shelf_structure.py
```

### Phase 3 - CLIP features
Each detected product gets cropped out and passed through CLIP ViT-B/32 to get a 512 dim embedding. Then we cluster visually similar products using cosine similarity.

```bash
python3 src/feature_encoding.py
```

### Phase 4 - Planogram compliance
We don't actually have a real planogram for SKU-110K so we generate an ideal one assuming uniform spacing within each row. Then we match detected products to ideal slots and count how many are off.

```bash
python3 src/planogram.py
```

## Running everything at once

```bash
python3 src/pipeline.py
```

Output goes to `results/pipeline_output/`. Or just use the HF space if don't want to install anything.

## Evaluation

```bash
python3 src/evaluate.py
```

This runs on test images and prints metrics and saves them to `results/evaluation_results.json` and a plot to `results/evaluation_plot.png`.

## Results

Training (10 epochs, image size 416):

| Epoch | box_loss | mAP50 | mAP50-95 |
|-------|----------|-------|----------|
| 1 | 2.138 | 0.282 | 0.096 |
| 5 | 1.734 | 0.536 | 0.257 |
| 10 | 1.637 | 0.616 | 0.322 |
| Best | - | 0.641 | 0.334 |


Evaluation on test images:

| Metric | Score |
|--------|-------|
| Avg Precision | 0.791 |
| Avg Recall | 0.674 |
| Avg F1 | 0.722 |
| Avg Planogram Compliance | 60.6% |
| Total OOS Gaps Detected | 83 |

Sample output for test_588.jpg:
- Phase 1: 183 products detected
- Phase 2: 9 rows, 1 OOS gap
- Phase 3: 30 embeddings generated
- Phase 4: 79% compliance, 37 violations

## Metrics we used

- Precision, Recall, F1 against ground truth at IoU 0.5
- mAP50 and mAP50-95 from Ultralytics
- Planogram Compliance=fraction of products within 8% normalized distance of their ideal position
- OOS Recall=number of empty gaps detected per image

## Why we chose what we chose

| Choice | Why |
|--------|-----|
| YOLOv11n instead of R-CNN | Faster, works better on dense scenes, smaller model |
| Standard bounding boxes | SKU-110K labels are already axis aligned so OBB would not help |
| CLIP ViT-B/32 | We did not want to fine tune another model, CLIP works zero-shot |
| Y-centroid clustering for rows | Simple and works, did not need extra annotations |
| Uniform spacing for ideal planogram | No real planogram in the dataset so we made one up |
| HF Spaces for demo | Free, public URL, zero installation |

## Fallback plan

If the full pipeline did not have worked out we were going to fall back to just 2 classes:
- Product (in stock)
- Empty Shelf (out of stock)

Same YOLO model just with simpler outputs. We did not ended up needing this since the full pipeline works.

## References

1. Real-time Planogram Compliance Application using Computer Vision and Virtual Shelves - Scientific Reports, 2025
2. Context-Aware Fine-Grained Product Recognition on Grocery Shelves - IEEE Access, 2025
3. Precise Detection in Densely Packed Scenes - CVPR (EM-merger)
4. SKU-110K Dataset - Goldman et al.,CVPR 2019
5. CLIP: Learning Transferable Visual Models from Natural Language Supervision - Radford et al.,2021

## Originality

This is original work for this class. We did not reuse code from any other project or coursework. YOLOv11, OpenCV and CLIP are used as pretrained components but the full pipeline (shelf structure analysis, planogram scoring, HF deployment, evaluation) is all written by us.