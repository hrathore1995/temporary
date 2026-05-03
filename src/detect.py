import os
import cv2
import torch
from ultralytics import YOLO
base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weights=os.path.join(base_dir, "results", "retailaudit_yolo11", "weights", "best.pt")
test_dir=os.path.join(base_dir, "data", "raw", "SKU110K_fixed", "images", "test")
output_dir=os.path.join(base_dir, "results", "detections")
os.makedirs(output_dir,exist_ok=True)
device="mps" if torch.backends.mps.is_available() else "cpu"
conf_threshold=0.3
iou_threshold=0.45
num_images=10
def run_detection():
    print(f"Loading model from: {weights}")
    model=YOLO(weights)
    test_images=[
        os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith((".jpg", ".jpeg", ".png"))][:num_images]
    print(f"Running detection on {len(test_images)} images")
    for img_path in test_images:
        img_name=os.path.basename(img_path)
        results=model.predict(
            source=img_path,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False)
        result=results[0]
        boxes=result.boxes
        num_detections=len(boxes)
        annotated=result.plot()
        out_path=os.path.join(output_dir, f"detected_{img_name}")
        cv2.imwrite(out_path, annotated)
        print(f"{img_name} -> {num_detections} products detected and saved to {out_path}")
    print(f"\nAll detections saved to: {output_dir}")

if __name__ == "__main__":
    print("RETAILAUDIT-NET DETECTION")
    run_detection()