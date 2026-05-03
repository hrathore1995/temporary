import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import clip
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weights=os.path.join(base_dir,"results","retailaudit_yolo11","weights","best.pt")
test_dir=os.path.join(base_dir,"data","raw","SKU110K_fixed","images","test")
label_dir=os.path.join(base_dir,"data","raw","SKU110K_fixed","labels","test")
output_dir=os.path.join(base_dir,"results")
os.makedirs(output_dir, exist_ok=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
conf_threshold=0.3
iou_threshold=0.45
gap_threshold=0.06
row_threshold_ratio=0.05
num_eval_images=20
def load_ground_truth(label_path,img_w,img_h):
    boxes=[]
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts=line.strip().split()
            if len(parts) < 5:
                continue
            cx,cy,bw,bh = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
            x1=(cx-bw/2)*img_w
            y1=(cy-bh/2) *img_h
            x2=(cx+bw/2)*img_w
            y2=(cy+bh/2) *img_h
            boxes.append([x1,y1,x2,y2])
    return np.array(boxes)
def iou(box1,box2):
    x1 = max(box1[0],box2[0])
    y1= max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 =min(box1[3],box2[3])
    inter =max(0,x2-x1)*max(0,y2-y1)
    area1 =(box1[2]-box1[0])*(box1[3]-box1[1])
    area2  = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union=area1+area2-inter
    return inter/union if union>0 else 0.0
def compute_detection_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(gt_boxes)==0:
        return 0.0,0.0,0.0
    if len(pred_boxes)==0:
        return 0.0,0.0,0.0
    matched_gt=set()
    tp=0
    for pred in pred_boxes:
        best_iou=0
        best_gt_idx=-1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou=iou(pred, gt)
            if iou>best_iou:
                best_iou=iou
                best_gt_idx=gt_idx
        if best_iou>=iou_threshold and best_gt_idx!=-1:
            tp+=1
            matched_gt.add(best_gt_idx)
    fp=len(pred_boxes)-tp
    fn=len(gt_boxes)-tp
    precision=tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1=2*precision*recall/ (precision+recall) if (precision+recall)>0 else 0.0
    return round(precision,3),round(recall,3), round(f1,3)
def assign_rows(boxes, img_height):
    if len(boxes)==0:
        return {},[]
    centers_y=np.array([(b[1] + b[3]) / 2 for b in boxes])
    sorted_y= np.sort(centers_y)
    threshold=img_height*row_threshold_ratio
    rows=[]
    current=[sorted_y[0]]
    for y in sorted_y[1:]:
        if abs(y-np.mean(current))<threshold:
            current.append(y)
        else:
            rows.append(current)
            current= [y]
    rows.append(current)
    row_means=[np.mean(r) for r in rows]
    assigned={i: [] for i in range(len(row_means))}
    for idx,box in enumerate(boxes):
        cy=(box[1]+box[3])/2
        row_idx=int(np.argmin([abs(cy-rm) for rm in row_means]))
        assigned[row_idx].append(idx)
    return assigned,row_means
def detect_gaps(boxes,assigned_rows,row_means,img_width,img_height):
    gaps=[]
    for row_idx, box_indices in assigned_rows.items():
        if len(box_indices)<2:
            continue
        row_boxes=sorted([boxes[i] for i in box_indices],key=lambda b: b[0])
        row_y=int(row_means[row_idx])
        avg_h=int(np.mean([b[3]-b[1] for b in row_boxes]))
        for i in range(len(row_boxes)- 1):
            gap_ratio=(row_boxes[i+1][0]-row_boxes[i][2])/img_width
            if gap_ratio>gap_threshold:
                gaps.append({
                    "row": row_idx,
                    "x1":int(row_boxes[i][2]),
                    "x2":int(row_boxes[i+1][0]),
                    "y1":max(0,row_y-avg_h//2),
                    "y2":min(img_height, row_y + avg_h//2),
                    "gap_ratio":round(gap_ratio, 3)
                })
    return gaps
def compute_planogram_score(boxes,assigned_rows,row_means,img_width,img_height):
    if not assigned_rows:
        return 0.0,[]
    violations=[]
    matched=0
    total=0
    for row_idx, box_indices in assigned_rows.items():
        if len(box_indices)==0:
            continue
        row_boxes=sorted([boxes[i] for i in box_indices],key=lambda b: b[0])
        num_items=len(row_boxes)
        spacing=img_width/(num_items+1)
        row_y=row_means[row_idx]
        avg_w=np.mean([b[2]- b[0] for b in row_boxes])
        for pos,box in enumerate(row_boxes):
            ideal_cx = spacing*(pos + 1)
            real_cx=(box[0] + box[2]) / 2
            dist=abs(real_cx - ideal_cx) / img_width
            total+=1
            if dist<0.08:
                matched+=1
            else:
                violations.append({"row": row_idx, "col": pos, "dist": round(dist, 3)})
    score = matched / total if total > 0 else 0.0
    return round(score, 3), violations
def plot_results(all_metrics, output_path):
    images=[m["image"].replace(".jpg", "") for m in all_metrics]
    precisions=[m["precision"] for m in all_metrics]
    recalls=[m["recall"] for m in all_metrics]
    compliances=[m["compliance"] for m in all_metrics]
    oos_gaps=[m["oos_gaps"] for m in all_metrics]
    products=[m["products"] for m in all_metrics]

    fig, axes=plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RetailAudit-Net Evaluation Results", fontsize=16, fontweight="bold")

    x=np.arange(len(images))
    width=0.35
    axes[0,0].bar(x-width/2,precisions,width,label="Precision",color="#2196F3")
    axes[0,0].bar(x+width/2,recalls,width,label="Recall",color="#4CAF50")
    axes[0,0].set_title("Detection Precision & Recall")
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(images,rotation=45,ha="right",fontsize=7)
    axes[0,0].set_ylim(0,1.0)
    axes[0,0].legend()
    axes[0, 0].grid(axis="y",alpha=0.3)
    bar_colors=["#4CAF50" if c >= 70 else "#FF9800" if c >= 40 else "#F44336" for c in compliances]
    axes[0,1].bar(images, compliances, color=bar_colors)
    axes[0,1].set_title("Planogram Compliance Score (%)")
    axes[0,1].set_xticks(range(len(images)))
    axes[0,1].set_xticklabels(images,rotation=45,ha="right",fontsize=7)
    axes[0,1].set_ylim(0,100)
    axes[0,1].axhline(y=70,color="red",linestyle="--",alpha=0.5,label="70% threshold")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y",alpha=0.3)
    for i, v in enumerate(compliances):
        axes[0,1].text(i,v+1,f"{v}%",ha="center",fontsize=7)
    axes[1,0].bar(images,oos_gaps,color="#F44336")
    axes[1,0].set_title("Out-of-Stock Gaps Detected")
    axes[1,0].set_xticks(range(len(images)))
    axes[1,0].set_xticklabels(images, rotation=45,ha="right",fontsize=7)
    axes[1,0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(oos_gaps):
        axes[1,0].text(i,v+0.05,str(v),ha="center",fontsize=8)
    axes[1,1].scatter(products,compliances,color="#9C27B0",s=80,zorder=5)
    for i, img in enumerate(images):
        axes[1,1].annotate(img, (products[i], compliances[i]),textcoords="offset points", xytext=(5, 5), fontsize=6)
    axes[1,1].set_title("Product Count vs Compliance")
    axes[1,1].set_xlabel("Number of Products Detected")
    axes[1,1].set_ylabel("Compliance Score (%)")
    axes[1,1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")
def run_evaluation():
    print("RETAILAUDIT-NET FULL EVALUATION")
    model=YOLO(weights)
    clip_model, preprocess=clip.load("ViT-B/32",device=device)
    print("Models loaded.")
    test_images=[
        os.path.join(test_dir,f)
        for f in sorted(os.listdir(test_dir))
        if f.endswith((".jpg", ".jpeg", ".png"))
    ][:num_eval_images]
    all_metrics=[]
    for img_path in test_images:
        img_name=os.path.basename(img_path)
        label_path=os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
        image=cv2.imread(img_path)
        h,w=image.shape[:2]
        results = model.predict(
            source=img_path,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False)
        pred_boxes=results[0].boxes.xyxy.cpu().numpy()
        gt_boxes=load_ground_truth(label_path,w,h)
        precision,recall,f1=compute_detection_metrics(pred_boxes,gt_boxes)
        assigned_rows,row_means=assign_rows(pred_boxes,h)
        gaps=detect_gaps(pred_boxes,assigned_rows,row_means,w,h)
        score,violations=compute_planogram_score(pred_boxes,assigned_rows,row_means,w,h)
        metrics = {
            "image": img_name,
            "products": len(pred_boxes),
            "gt_products": len(gt_boxes),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rows": len(row_means),
            "oos_gaps": len(gaps),
            "compliance": int(score * 100),
            "violations": len(violations)
        }
        all_metrics.append(metrics)
        print(f"{img_name}: P={precision:.2f} R={recall:.2f} F1={f1:.2f} | "f"OOS={len(gaps)} | Compliance={int(score*100)}%")
    avg_precision=np.mean([m["precision"] for m in all_metrics])
    avg_recall=np.mean([m["recall"] for m in all_metrics])
    avg_f1=np.mean([m["f1"] for m in all_metrics])
    avg_compliance=np.mean([m["compliance"] for m in all_metrics])
    total_oos=sum(m["oos_gaps"] for m in all_metrics)
    print("AGGREGATE METRICS")
    print(f"Avg Precision: {avg_precision:.3f}")
    print(f"Avg Recall: {avg_recall:.3f}")
    print(f"Avg F1 Score: {avg_f1:.3f}")
    print(f"Avg Planogram Compliance: {avg_compliance:.1f}%")
    print(f"Total OOS Gaps Found: {total_oos}")
    print(f"Images Evaluated: {len(all_metrics)}")
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "summary": {
                "avg_precision": round(avg_precision, 3),
                "avg_recall": round(avg_recall, 3),
                "avg_f1": round(avg_f1, 3),
                "avg_compliance": round(avg_compliance, 1),
                "total_oos_gaps": total_oos,
                "images_evaluated": len(all_metrics)
            },
            "per_image": all_metrics
        }, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    plot_path = os.path.join(output_dir, "evaluation_plot.png")
    plot_results(all_metrics, plot_path)
if __name__ == "__main__":
    run_evaluation()