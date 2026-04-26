import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import clip

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(BASE_DIR, "results", "retailaudit_yolo11", "weights", "best.pt")
TEST_DIR = os.path.join(BASE_DIR, "data", "raw", "SKU110K_fixed", "images", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "planogram")
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
        box_heights = [(b[3] - b[1]) for b in row_boxes]
        avg_h = int(np.mean(box_heights)) if box_heights else 40

        for i in range(len(row_boxes) - 1):
            right_edge = row_boxes[i][2]
            left_edge = row_boxes[i + 1][0]
            gap_ratio = (left_edge - right_edge) / img_width
            if gap_ratio > GAP_THRESHOLD:
                gaps.append({
                    "row": row_idx,
                    "x1": int(right_edge),
                    "x2": int(left_edge),
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

def build_graph(boxes, assigned_rows, embeddings, valid_indices):
    idx_to_emb = {valid_indices[i]: embeddings[i] for i in range(len(valid_indices))}
    nodes = []
    for row_idx, box_indices in assigned_rows.items():
        row_boxes = sorted(box_indices, key=lambda i: boxes[i][0])
        for pos, box_idx in enumerate(row_boxes):
            nodes.append({
                "box_idx": box_idx,
                "row": row_idx,
                "col": pos,
                "box": boxes[box_idx],
                "embedding": idx_to_emb.get(box_idx, None)
            })
    return nodes

def generate_ideal_planogram(nodes, assigned_rows, row_means, img_width, img_height):
    ideal = []
    for row_idx, box_indices in assigned_rows.items():
        if len(box_indices) == 0:
            continue
        num_items = len(box_indices)
        spacing = img_width / (num_items + 1)
        row_y = row_means[row_idx]
        row_node_boxes = [nodes[i]["box"] for i in range(len(nodes)) if nodes[i]["row"] == row_idx]
        avg_w = np.mean([b[2] - b[0] for b in row_node_boxes]) if row_node_boxes else 60
        avg_h = np.mean([b[3] - b[1] for b in row_node_boxes]) if row_node_boxes else 60

        for pos in range(num_items):
            cx = spacing * (pos + 1)
            ideal.append({
                "row": row_idx,
                "col": pos,
                "cx": cx,
                "cy": row_y,
                "w": avg_w,
                "h": avg_h
            })
    return ideal

def compute_planogram_score(nodes, ideal_nodes, img_width, img_height):
    if not nodes or not ideal_nodes:
        return 0.0, []

    violations = []
    matched = 0

    real_by_pos = {(n["row"], n["col"]): n for n in nodes}
    ideal_by_pos = {(n["row"], n["col"]): n for n in ideal_nodes}

    for pos_key, ideal in ideal_by_pos.items():
        real = real_by_pos.get(pos_key)
        if real is None:
            violations.append({"type": "missing", "row": ideal["row"], "col": ideal["col"]})
            continue

        real_cx = (real["box"][0] + real["box"][2]) / 2
        real_cy = (real["box"][1] + real["box"][3]) / 2
        dist = np.sqrt((real_cx - ideal["cx"])**2 + (real_cy - ideal["cy"])**2)
        norm_dist = dist / np.sqrt(img_width**2 + img_height**2)

        if norm_dist < 0.08:
            matched += 1
        else:
            violations.append({
                "type": "misplaced",
                "row": ideal["row"],
                "col": ideal["col"],
                "dist": round(norm_dist, 3)
            })

    score = matched / len(ideal_by_pos) if ideal_by_pos else 0.0
    return round(score, 3), violations

def draw_full_analysis(image, boxes, assigned_rows, row_means, gaps, nodes, ideal_nodes, score, violations):
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
        cv2.putText(output, f"R{row_idx+1}",
                    (5, row_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        for idx in box_indices:
            x1, y1, x2, y2 = map(int, boxes[idx])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

    for gap in gaps:
        overlay = output.copy()
        cv2.rectangle(overlay, (gap["x1"], gap["y1"]), (gap["x2"], gap["y2"]), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.45, output, 0.55, 0, output)
        cx = (gap["x1"] + gap["x2"]) // 2
        cy = (gap["y1"] + gap["y2"]) // 2
        cv2.putText(output, "OOS", (cx - 15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    for ideal in ideal_nodes:
        cx, cy = int(ideal["cx"]), int(ideal["cy"])
        hw, hh = int(ideal["w"] / 2), int(ideal["h"] / 2)
        cv2.rectangle(output, (cx - hw, cy - hh), (cx + hw, cy + hh), (0, 255, 200), 1)

    score_color = (0, 220, 0) if score >= 0.7 else (0, 140, 255) if score >= 0.4 else (0, 0, 255)
    compliance_pct = int(score * 100)

    cv2.rectangle(output, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.putText(output, f"RetailAudit-Net | Products: {len(boxes)} | Rows: {len(row_means)} | Gaps(OOS): {len(gaps)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(output, f"Planogram Compliance: {compliance_pct}% | Violations: {len(violations)}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)

    return output

def process_image(yolo_model, clip_model, preprocess, img_path):
    img_name = os.path.basename(img_path)
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    print(f"\nProcessing: {img_name} ({w}x{h})")

    boxes, confs = detect_products(yolo_model, img_path)
    print(f"  Products detected: {len(boxes)}")

    assigned_rows, row_means = assign_rows(boxes, h)
    print(f"  Rows identified: {len(row_means)}")

    gaps = detect_gaps(boxes, assigned_rows, row_means, w, h)
    print(f"  OOS gaps detected: {len(gaps)}")

    embeddings, valid_indices = encode_crops(clip_model, preprocess, image, boxes)
    print(f"  CLIP embeddings: {len(embeddings)}")

    nodes = build_graph(boxes, assigned_rows, embeddings, valid_indices)
    ideal_nodes = generate_ideal_planogram(nodes, assigned_rows, row_means, w, h)

    score, violations = compute_planogram_score(nodes, ideal_nodes, w, h)
    print(f"  Planogram compliance score: {int(score*100)}%")
    print(f"  Violations: {len(violations)}")

    output = draw_full_analysis(image, boxes, assigned_rows, row_means,
                                gaps, nodes, ideal_nodes, score, violations)

    out_path = os.path.join(OUTPUT_DIR, f"planogram_{img_name}")
    cv2.imwrite(out_path, output)
    print(f"  Saved: {out_path}")

    return {
        "image": img_name,
        "products": len(boxes),
        "rows": len(row_means),
        "oos_gaps": len(gaps),
        "compliance": int(score * 100),
        "violations": len(violations)
    }

def run_planogram():
    print("=" * 50)
    print("PHASE 4: PLANOGRAM COMPLIANCE")
    print("=" * 50)

    yolo_model = YOLO(WEIGHTS)
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    print("Models loaded.")

    test_images = [
        os.path.join(TEST_DIR, f)
        for f in os.listdir(TEST_DIR)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ][:5]

    all_results = []
    for img_path in test_images:
        result = process_image(yolo_model, clip_model, preprocess, img_path)
        all_results.append(result)

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"{'Image':<20} {'Products':>8} {'Rows':>6} {'OOS':>5} {'Compliance':>12} {'Violations':>11}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['image']:<20} {r['products']:>8} {r['rows']:>6} {r['oos_gaps']:>5} {r['compliance']:>11}% {r['violations']:>11}")

if __name__ == "__main__":
    run_planogram()