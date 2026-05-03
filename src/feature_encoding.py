import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weights=os.path.join(base_dir,"results","retailaudit_yolo11","weights","best.pt")
test_dir=os.path.join(base_dir,"data","raw","SKU110K_fixed","images","test")
output_dir=os.path.join(base_dir,"results","planogram")
os.makedirs(output_dir, exist_ok=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
conf_threshold=0.3
iou_threshold=0.45
def install_clip():
    try:
        import clip
        return clip
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install","git+https://github.com/openai/CLIP.git"])
        import clip
        return clip
def load_clip_model():
    clip=install_clip()
    model,preprocess=clip.load("ViT-B/32", device=device)
    return model,preprocess,clip
def detect_products(model,img_path):
    results = model.predict(
        source=img_path,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=False)
    boxes=results[0].boxes.xyxy.cpu().numpy()
    return boxes
def crop_product(image,box,padding=5):
    h,w=image.shape[:2]
    x1=max(0,int(box[0])-padding)
    y1=max(0,int(box[1])-padding)
    x2=min(w,int(box[2])+padding)
    y2=min(h,int(box[3])+padding)
    crop=image[y1:y2,x1:x2]
    return crop
def encode_products(clip_model,preprocess,clip,image, boxes,max_crops=20):
    embeddings=[]
    selected_boxes=boxes[:max_crops]
    for box in selected_boxes:
        crop=crop_product(image, box)
        if crop.size == 0:
            continue
        crop_rgb=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img=Image.fromarray(crop_rgb)
        tensor=preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding=clip_model.encode_image(tensor)
            embedding=embedding/embedding.norm(dim=-1, keepdim=True)
            embeddings.append(embedding.cpu().numpy()[0])
    return np.array(embeddings), selected_boxes
def compute_similarity_matrix(embeddings):
    if len(embeddings) < 2:
        return np.array([])
    sim_matrix=np.dot(embeddings, embeddings.T)
    return sim_matrix
def cluster_products(embeddings, boxes, sim_threshold=0.85):
    if len(embeddings) < 2:
        return {0:list(range(len(embeddings)))}
    sim_matrix=compute_similarity_matrix(embeddings)
    clusters={}
    assigned=set()
    cluster_id=0
    for i in range(len(embeddings)):
        if i in assigned:
            continue
        cluster=[i]
        assigned.add(i)
        for j in range(i+1,len(embeddings)):
            if j not in assigned and sim_matrix[i][j]>sim_threshold:
                cluster.append(j)
                assigned.add(j)
        clusters[cluster_id]=cluster
        cluster_id+=1
    return clusters
def draw_clusters(image, boxes, clusters):
    output = image.copy()
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,165,0),(128,0,128),(0,255,255),(255,20,147),(0,128,0),(255,255,0),(0,165,255)]
    box_to_cluster = {}
    for cluster_id, box_indices in clusters.items():
        for idx in box_indices:
            box_to_cluster[idx]=cluster_id
    for idx,box in enumerate(boxes):
        x1,y1,x2,y2=map(int, box)
        cluster_id=box_to_cluster.get(idx, 0)
        color=colors[cluster_id%len(colors)]
        cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
        cv2.putText(output,f"C{cluster_id}",(x1,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.3,color,1)
    return output
def process_image(yolo_model,clip_model,preprocess,clip,img_path):
    img_name=os.path.basename(img_path)
    image=cv2.imread(img_path)
    h,w=image.shape[:2]
    print(f"\nProcessing: {img_name} ({w}x{h})")
    boxes=detect_products(yolo_model,img_path)
    print(f"Products detected: {len(boxes)}")
    embeddings,selected_boxes=encode_products(clip_model,preprocess,clip,image,boxes)
    print(f"Products encoded with CLIP: {len(embeddings)}")
    if len(embeddings)>1:
        sim_matrix=compute_similarity_matrix(embeddings)
        avg_similarity=(sim_matrix.sum()-len(embeddings))/(len(embeddings)*(len(embeddings)-1))
        print(f"Avg pairwise similarity: {avg_similarity:.3f}")
        clusters=cluster_products(embeddings, selected_boxes)
        print(f"Product clusters found: {len(clusters)}")
    else:
        clusters={0:[0]} if len(embeddings)==1 else {}
        print("Not enough products to cluster.")
    output=draw_clusters(image,selected_boxes,clusters)
    num_clusters=len(clusters)
    cv2.putText(output,f"Products: {len(boxes)} | Encoded: {len(embeddings)} | Clusters: {num_clusters}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    out_path=os.path.join(output_dir,f"clip_{img_name}")
    cv2.imwrite(out_path,output)
    print(f"Saved to: {out_path}")
    return {
        "image": img_name,
        "products": len(boxes),
        "encoded": len(embeddings),
        "clusters": num_clusters
    }
def run_feature_encoding():
    print("PHASE 3: CLIP FEATURE ENCODING")
    print("Loading CLIP model (downloading if first time)...")
    clip_model, preprocess, clip = load_clip_model()
    print("CLIP model loaded.")
    yolo_model = YOLO(weights)
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith((".jpg", ".jpeg", ".png"))][:5]
    all_results = []
    for img_path in test_images:
        result = process_image(yolo_model, clip_model, preprocess, clip, img_path)
        all_results.append(result)
    print("\nSUMMARY")
    for r in all_results:
        print(f"{r['image']}: {r['products']} products | {r['encoded']} encoded | {r['clusters']} clusters")
if __name__ == "__main__":
    run_feature_encoding()