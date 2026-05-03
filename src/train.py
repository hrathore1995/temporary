import os
import torch
from ultralytics import YOLO
base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_yaml=os.path.join(base_dir,"data","sku110k.yaml")
weights_dir=os.path.join(base_dir,"models","weights")
os.makedirs(weights_dir,exist_ok=True)
device="mps" if torch.backends.mps.is_available() else "cpu"
model="yolo11n.pt"
EPOCHS=10
batch_size=32
img_size=416
project=os.path.join(base_dir,"results")
RUN_NAME="retailaudit_yolo11"
def train():
    print(f"device: {device}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {batch_size}")
    model=YOLO(model)
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
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
        fraction=0.2)
    print("Training complete.")
    print(f"Best weights saved to: {project}/{RUN_NAME}/weights/best.pt")
if __name__ == "__main__":
    print("RETAILAUDIT-NET TRAINING")
    train()