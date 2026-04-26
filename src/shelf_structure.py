import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(BASE_DIR, "results", "retailaudit_yolo11", "weights", "best.pt")
TEST_DIR = os.path.join(BASE_DIR, "data", "raw", "SKU110K_fixed", "images", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "planogram")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45

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

def detect_shelf_rows(image, boxes):
    if len(boxes) == 0:
        return [], image.copy()

    centers_y = [(box[1] + box[3]) / 2 for box in boxes]
    centers_y = np.array(centers_y)
    centers_y_sorted = np.sort(centers_y)

    rows = []
    current_row = [centers_y_sorted[0]]
    row_threshold = image.shape[0] * 0.05

    for y in centers_y_sorted[1:]:
        if abs(y - np.mean(current_row)) < row_threshold:
            current_row.append(y)
        else:
            rows.append(current_row)
            current_row = [y]
    rows.append(current_row)

    row_means = [np.mean(r) for r in rows]
    return row_means, len(rows)

def normalize_perspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=image.shape[1] * 0.3,
        maxLineGap=20
    )

    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 15 or angle > 165:
                horizontal_lines.append((x1, y1, x2, y2))

    return horizontal_lines, edges

def assign_boxes_to_rows(boxes, row_means, img_height):
    row_threshold = img_height * 0.05
    assigned = {i: [] for i in range(len(row_means))}

    for box in boxes:
        center_y = (box[1] + box[3]) / 2
        distances = [abs(center_y - rm) for rm in row_means]
        closest_row = np.argmin(distances)
        assigned[closest_row].append(box)

    return assigned

def draw_shelf_analysis(image, boxes, row_means, horizontal_lines, assigned_rows):
    output = image.copy()
    h, w = output.shape[:2]

    for x1, y1, x2, y2 in horizontal_lines:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 1)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 165, 0), (128, 0, 128), (0, 255, 255),
        (255, 20, 147), (0, 128, 0)
    ]

    for row_idx, row_boxes in assigned_rows.items():
        color = colors[row_idx % len(colors)]
        row_y = int(row_means[row_idx])
        cv2.line(output, (0, row_y), (w, row_y), color, 2)
        cv2.putText(output, f"Row {row_idx + 1} ({len(row_boxes)} items)",
                    (10, row_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for box in row_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

    return output

def analyze_shelf_gaps(assigned_rows, img_width, gap_threshold=0.08):
    gaps = {}
    for row_idx, row_boxes in assigned_rows.items():
        if len(row_boxes) < 2:
            continue

        row_boxes_sorted = sorted(row_boxes, key=lambda b: b[0])
        row_gaps = []

        for i in range(len(row_boxes_sorted) - 1):
            right_edge = row_boxes_sorted[i][2]
            left_edge = row_boxes_sorted[i + 1][0]
            gap_size = (left_edge - right_edge) / img_width

            if gap_size > gap_threshold:
                gap_center_x = int((right_edge + left_edge) / 2)
                row_gaps.append({
                    "gap_start": int(right_edge),
                    "gap_end": int(left_edge),
                    "gap_center_x": gap_center_x,
                    "gap_size_ratio": round(gap_size, 3)
                })

        if row_gaps:
            gaps[row_idx] = row_gaps

    return gaps

def draw_gaps(output, gaps, row_means, img_height):
    for row_idx, row_gaps in gaps.items():
        row_y = int(row_means[row_idx])
        for gap in row_gaps:
            x1 = gap["gap_start"]
            x2 = gap["gap_end"]
            y1 = max(0, row_y - 40)
            y2 = min(img_height, row_y + 40)
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
            cv2.putText(output, "EMPTY", (x1 + 2, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return output

def process_image(model, img_path):
    img_name = os.path.basename(img_path)
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    print(f"\nProcessing: {img_name} ({w}x{h})")

    boxes, confs = detect_products(model, img_path)
    print(f"  Products detected: {len(boxes)}")

    horizontal_lines, edges = normalize_perspective(image)
    print(f"  Horizontal shelf lines found: {len(horizontal_lines)}")

    row_means, num_rows = detect_shelf_rows(image, boxes)
    print(f"  Shelf rows identified: {num_rows}")

    assigned_rows = assign_boxes_to_rows(boxes, row_means, h)

    gaps = analyze_shelf_gaps(assigned_rows, w)
    total_gaps = sum(len(g) for g in gaps.values())
    print(f"  Empty gaps detected: {total_gaps}")

    output = draw_shelf_analysis(image, boxes, row_means, horizontal_lines, assigned_rows)
    output = draw_gaps(output, gaps, row_means, h)

    cv2.putText(output, f"Products: {len(boxes)} | Rows: {num_rows} | Gaps: {total_gaps}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, f"Horizontal lines: {len(horizontal_lines)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = os.path.join(OUTPUT_DIR, f"shelf_{img_name}")
    cv2.imwrite(out_path, output)
    print(f"  Saved to: {out_path}")

    return {
        "image": img_name,
        "products": len(boxes),
        "rows": num_rows,
        "gaps": total_gaps,
        "shelf_lines": len(horizontal_lines)
    }

def run_shelf_analysis():
    print("=" * 50)
    print("PHASE 2: SHELF STRUCTURE ANALYSIS")
    print("=" * 50)

    model = YOLO(WEIGHTS)

    test_images = [
        os.path.join(TEST_DIR, f)
        for f in os.listdir(TEST_DIR)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ][:5]

    all_results = []
    for img_path in test_images:
        result = process_image(model, img_path)
        all_results.append(result)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for r in all_results:
        print(f"{r['image']}: {r['products']} products | {r['rows']} rows | {r['gaps']} gaps")

if __name__ == "__main__":
    run_shelf_analysis()