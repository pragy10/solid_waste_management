
import os
from pathlib import Path
from collections import Counter
import yaml

TARGET_CLASSES = ['rigid_plastic', 'soft_plastic', 'glass', 'metal', 'cardboard', 'paper']

def analyze_yolo_dataset(dataset_path):
    """Complete analytics for YOLO dataset"""
    dataset_path = Path(dataset_path)
    print(f"\n{'='*80}")
    print(f"swm DATASET ANALYTICS: {dataset_path}")
    print(f"{'='*80}")
    
    
    train_imgs = len(list((dataset_path / "train/images").rglob("*.jpg")))
    train_lbls = len(list((dataset_path / "train/labels").rglob("*.txt")))
    val_imgs = len(list((dataset_path / "val/images").rglob("*.jpg")))
    val_lbls = len(list((dataset_path / "val/labels").rglob("*.txt")))
    
    print(f"Train Images: {train_imgs:>4} | Train Labels: {train_lbls:>4}")
    print(f"Val Images:   {val_imgs:>4} | Val Labels:   {val_lbls:>4}")
    print(f"Total Images: {train_imgs + val_imgs:>4} | Total Labels: {train_lbls + val_lbls:>4}")
    
    
    print("\nCLASS DISTRIBUTION (All instances):")
    class_counts = Counter()
    
    for split in ['train', 'val']:
        split_path = dataset_path / split / "labels"
        for txt_file in split_path.rglob("*.txt"):
            with open(txt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        cls_id = int(parts[0])
                        class_counts[cls_id] += 1
    
    print("Class ID | Count     | Name             | %")
    print("-" * 45)
    total_instances = sum(class_counts.values())
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        name = TARGET_CLASSES[cls_id] if cls_id < len(TARGET_CLASSES) else f"UNKNOWN_{cls_id}"
        pct = (count / total_instances * 100) if total_instances > 0 else 0
        print(f"{cls_id:>7} | {count:>8} | {name:<15} | {pct:>5.1f}%")
    
    
    print("\nPER-IMAGE STATISTICS:")
    img_with_labels = 0
    empty_labels = 0
    total_bboxes = 0
    
    for split in ['train', 'val']:
        split_path = dataset_path / split / "labels"
        for txt_file in split_path.rglob("*.txt"):
            bbox_count = 0
            with open(txt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        bbox_count += 1
            total_bboxes += bbox_count
            if bbox_count > 0:
                img_with_labels += 1
            else:
                empty_labels += 1
    
    total_images = train_imgs + val_imgs
    print(f"Images with ≥1 bbox: {img_with_labels:>4} ({img_with_labels/total_images*100:.1f}%)")
    print(f"Empty label files:    {empty_labels:>4} ({empty_labels/total_images*100:.1f}%)")
    print(f"Avg bboxes per image: {total_bboxes/total_images:.2f}")
    
    
    print("\nCLASS BALANCE:")
    max_count = max(class_counts.values())
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        ratio = max_count / count if count > 0 else float('inf')
        name = TARGET_CLASSES[cls_id] if cls_id < len(TARGET_CLASSES) else f"UNKNOWN_{cls_id}"
        status = "[✓]" if ratio < 5 else "[X]"
    
    
    yaml_path = Path("swm.yaml")
    if yaml_path.exists():
        print(f"\nYAML found: {yaml_path}")
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        print(f"   nc: {yaml_data.get('nc', 'N/A')} classes")
        print(f"   path: {yaml_data.get('path', 'N/A')}")
    
    
    print("\nSUMMARY TABLE")
    print("-" * 60)
    print(f"{'Split':<8} {'Images':>6} {'Labels':>6} {'Avg BBox':>8}")
    print(f"{'Train':<8} {train_imgs:>6} {train_lbls:>6} {train_lbls/train_imgs:>8.1f}")
    print(f"{'Val':<8} {val_imgs:>6} {val_lbls:>6} {val_lbls/val_imgs:>8.1f}")
    print(f"{'Total':<8} {total_images:>6} {total_bboxes:>6} {total_bboxes/total_images:>8.1f}")

if __name__ == "__main__":
    
    analyze_yolo_dataset("./swm_final")
