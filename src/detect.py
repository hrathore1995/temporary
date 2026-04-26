import os
import cv2
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(BASE_DIR, "results", "retailaudit_yolo11", "weights", "best.pt")
TEST_DIR = os.path.join(BASE_DIR, "data", "raw", "SKU110K_fixed", "images", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "detections")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
NUM_IMAGES = 10

def run_detection():
    print(f"Loading model from: {WEIGHTS}")
    model = YOLO(WEIGHTS)

    test_images = [
        os.path.join(TEST_DIR, f)
        for f in os.listdir(TEST_DIR)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ][:NUM_IMAGES]

    print(f"Running detection on {len(test_images)} images...")

    for img_path in test_images:
        img_name = os.path.basename(img_path)
        results = model.predict(
            source=img_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes
        num_detections = len(boxes)

        annotated = result.plot()
        out_path = os.path.join(OUTPUT_DIR, f"detected_{img_name}")
        cv2.imwrite(out_path, annotated)
        print(f"{img_name} -> {num_detections} products detected -> saved to {out_path}")

    print(f"\nAll detections saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    print("=" * 50)
    print("RETAILAUDIT-NET DETECTION")
    print("=" * 50)
    run_detection()