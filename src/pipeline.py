import os
import sys
import cv2
import torch
import numpy as np
import clip
from ultralytics import YOLO
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(BASE_DIR, "results", "retailaudit_yolo11", "weights", "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "pipeline_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
GAP_THRESHOLD = 0.06
ROW_THRESHOLD_RATIO = 0.05
MAX_ENCODE = 30

def detect_products(model, img_path):
    results = model.predict(
        source=img_path,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        verbose=False
    )
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    return boxes, confs

def assign_rows(boxes, img_height):
    if len(boxes) == 0:
        return {}, []
    centers_y = np.array([(b[1] + b[3]) / 2 for b in boxes])
    sorted_y = np.sort(centers_y)
    threshold = img_height * ROW_THRESHOLD_RATIO
    rows = []
    current = [sorted_y[0]]
    for y in sorted_y[1:]:
        if abs(y - np.mean(current)) < threshold:
            current.append(y)
        else:
            rows.append(current)
            current = [y]
    rows.append(current)
    row_means = [np.mean(r) for r in rows]
    assigned = {i: [] for i in range(len(row_means))}
    for idx, box in enumerate(boxes):
        cy = (box[1] + box[3]) / 2
        row_idx = int(np.argmin([abs(cy - rm) for rm in row_means]))
        assigned[row_idx].append(idx)
    return assigned, row_means

def detect_gaps(boxes, assigned_rows, row_means, img_width, img_height):
    gaps = []
    for row_idx, box_indices in assigned_rows.items():
        if len(box_indices) < 2:
            continue
        row_boxes = sorted([boxes[i] for i in box_indices], key=lambda b: b[0])
        row_y = int(row_means[row_idx])
        avg_h = int(np.mean([b[3] - b[1] for b in row_boxes]))
        for i in range(len(row_boxes) - 1):
            gap_ratio = (row_boxes[i+1][0] - row_boxes[i][2]) / img_width
            if gap_ratio > GAP_THRESHOLD:
                gaps.append({
                    "row": row_idx,
                    "x1": int(row_boxes[i][2]),
                    "x2": int(row_boxes[i+1][0]),
                    "y1": max(0, row_y - avg_h // 2),
                    "y2": min(img_height, row_y + avg_h // 2),
                    "gap_ratio": round(gap_ratio, 3)
                })
    return gaps

def encode_crops(clip_model, preprocess, image, boxes):
    embeddings = []
    valid_indices = []
    for idx, box in enumerate(boxes[:MAX_ENCODE]):
        x1, y1, x2, y2 = map(int, box)
        crop = image[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0:
            continue
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = preprocess(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy()[0])
            valid_indices.append(idx)
    return np.array(embeddings), valid_indices

def compute_planogram_score(boxes, assigned_rows, row_means, img_width, img_height):
    if not assigned_rows:
        return 0.0, []
    violations = []
    matched = 0
    total = 0
    for row_idx, box_indices in assigned_rows.items():
        if len(box_indices) == 0:
            continue
        row_boxes = sorted([boxes[i] for i in box_indices], key=lambda b: b[0])
        num_items = len(row_boxes)
        spacing = img_width / (num_items + 1)
        for pos, box in enumerate(row_boxes):
            ideal_cx = spacing * (pos + 1)
            real_cx = (box[0] + box[2]) / 2
            dist = abs(real_cx - ideal_cx) / img_width
            total += 1
            if dist < 0.08:
                matched += 1
            else:
                violations.append({"row": row_idx, "col": pos, "dist": round(dist, 3)})
    score = matched / total if total > 0 else 0.0
    return round(score, 3), violations

def draw_output(image, boxes, assigned_rows, row_means, gaps, score, violations, embeddings):
    output = image.copy()
    h, w = output.shape[:2]

    colors = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 200, 0), (0, 200, 255), (200, 0, 255),
        (255, 128, 0), (0, 255, 128)
    ]

    for row_idx, box_indices in assigned_rows.items():
        color = colors[row_idx % len(colors)]
        row_y = int(row_means[row_idx])
        cv2.line(output, (0, row_y), (w, row_y), color, 1)
        cv2.putText(output, f"Row {row_idx+1} ({len(box_indices)} items)",
                    (5, row_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        for idx in box_indices:
            x1, y1, x2, y2 = map(int, boxes[idx])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

    for gap in gaps:
        overlay = output.copy()
        cv2.rectangle(overlay, (gap["x1"], gap["y1"]), (gap["x2"], gap["y2"]), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.45, output, 0.55, 0, output)
        cx = (gap["x1"] + gap["x2"]) // 2
        cy = (gap["y1"] + gap["y2"]) // 2
        cv2.putText(output, "OOS", (cx - 15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    compliance_pct = int(score * 100)
    score_color = (0, 200, 0) if compliance_pct >= 70 else (0, 140, 255) if compliance_pct >= 40 else (0, 0, 255)

    cv2.rectangle(output, (0, 0), (w, 90), (15, 15, 15), -1)
    cv2.putText(output, "RetailAudit-Net | Full Pipeline",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(output, f"Phase 1: {len(boxes)} products detected | Phase 2: {len(row_means)} rows identified",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(output, f"Phase 3: {len(embeddings)} CLIP embeddings | Phase 4: Compliance {compliance_pct}% | OOS: {len(gaps)} | Violations: {len(violations)}",
                (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 1)

    return output

def run_pipeline(img_path):
    print("=" * 60)
    print("RETAILAUDIT-NET FULL PIPELINE")
    print("=" * 60)

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        sys.exit(1)

    img_name = os.path.basename(img_path)
    print(f"Input image: {img_name}")
    print(f"Device: {DEVICE}")

    print("\n[Phase 1] Loading YOLO model and detecting products...")
    yolo_model = YOLO(WEIGHTS)
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    boxes, confs = detect_products(yolo_model, img_path)
    print(f"  Products detected: {len(boxes)}")

    print("\n[Phase 2] Shelf structure and row segmentation...")
    assigned_rows, row_means = assign_rows(boxes, h)
    gaps = detect_gaps(boxes, assigned_rows, row_means, w, h)
    print(f"  Rows identified: {len(row_means)}")
    print(f"  OOS gaps detected: {len(gaps)}")

    print("\n[Phase 3] CLIP feature encoding...")
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    embeddings, valid_indices = encode_crops(clip_model, preprocess, image, boxes)
    print(f"  Embeddings generated: {len(embeddings)}")

    print("\n[Phase 4] Planogram compliance scoring...")
    score, violations = compute_planogram_score(boxes, assigned_rows, row_means, w, h)
    print(f"  Compliance score: {int(score*100)}%")
    print(f"  Violations: {len(violations)}")

    output = draw_output(image, boxes, assigned_rows, row_means,
                         gaps, score, violations, embeddings)

    out_path = os.path.join(OUTPUT_DIR, f"pipeline_{img_name}")
    cv2.imwrite(out_path, output)

    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(f"Products detected:      {len(boxes)}")
    print(f"Shelf rows:             {len(row_means)}")
    print(f"OOS gaps:               {len(gaps)}")
    print(f"CLIP embeddings:        {len(embeddings)}")
    print(f"Planogram compliance:   {int(score*100)}%")
    print(f"Violations:             {len(violations)}")
    print(f"Output saved to:        {out_path}")

    return out_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        test_dir = os.path.join(BASE_DIR, "data", "raw", "SKU110K_fixed", "images", "test")
        img_path = os.path.join(test_dir, "test_588.jpg")
        print(f"No image provided, using default: {img_path}")

    output_path = run_pipeline(img_path)
    print(f"\nDone. Open result with:")
    print(f"open {output_path}")