import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(BASE_DIR, "data", "sku110k.yaml")
WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL = "yolo11n.pt"
EPOCHS = 10
BATCH_SIZE = 32
IMG_SIZE = 416
PROJECT = os.path.join(BASE_DIR, "results")
RUN_NAME = "retailaudit_yolo11"

def train():
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 50)

    model = YOLO(MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        patience=10,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True,
        fraction=0.2
    )

    print("Training complete.")
    print(f"Best weights saved to: {PROJECT}/{RUN_NAME}/weights/best.pt")

if __name__ == "__main__":
    print("=" * 50)
    print("RETAILAUDIT-NET TRAINING")
    print("=" * 50)
    train()