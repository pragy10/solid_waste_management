import json
import cv2
from pathlib import Path


zerowaste_root = Path("./zerowaste/train")
json_file = zerowaste_root / "labels.json"
output_root = Path("./zerowaste_yolo")


ZW_MAP = {
    1: 0, 
    2: 4, 
    3: 3, 
    4: 1  
}

def coco_to_yolo(coco_bbox, img_width, img_height):
    """Convert COCO bbox [x, y, w, h] to YOLO [x_center, y_center, w, h] normalized"""
    x, y, w, h = coco_bbox
    
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    
    return [x_center, y_center, width, height]

def convert_zerowaste_to_yolo():
    """Convert ZeroWaste COCO JSON to YOLO format"""
    
    
    with open(json_file) as f:
        data = json.load(f)
    
    
    for split in ['train', 'val', 'test']:
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    
    converted = 0
    skipped = 0
    
    
    for img_info in data['images']:
        img_id = img_info['id']
        fname = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        
        img_path = zerowaste_root / "data" / fname
        if not img_path.exists():
            img_path = zerowaste_root / fname
        
        if not img_path.exists():
            print(f"Image not found: {fname}")
            skipped += 1
            continue
        
        
        anns = [a for a in data['annotations'] if a['image_id'] == img_id]
        
        
        yolo_lines = []
        for ann in anns:
            coco_cat_id = ann['category_id']
            
            
            if coco_cat_id not in ZW_MAP:
                continue  
            
            target_class = ZW_MAP[coco_cat_id]
            coco_bbox = ann['bbox']
            
            
            yolo_bbox = coco_to_yolo(coco_bbox, width, height)
            
            
            yolo_line = f"{target_class} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
            yolo_lines.append(yolo_line)
        
        
        
        split = 'train'  
        if 'split' in img_info:
            split = img_info['split']
        
        
        import shutil
        dest_img = output_root / split / 'images' / fname
        shutil.copy(img_path, dest_img)
        
        
        label_fname = Path(fname).stem + '.txt'
        label_path = output_root / split / 'labels' / label_fname
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        converted += 1
        if converted % 500 == 0:
            print(f"Converted {converted} images...")
    
    print(f"\n Conversion complete!")
    print(f"   Converted: {converted} images")
    print(f"   Skipped: {skipped} images")
    print(f"   Output: {output_root}")


convert_zerowaste_to_yolo()


print("\n" + "="*50)
print("VERIFICATION:")
print("="*50)

train_images = list((output_root / 'train' / 'images').glob('*'))
train_labels = list((output_root / 'train' / 'labels').glob('*.txt'))

print(f"Train images: {len(train_images)}")
print(f"Train labels: {len(train_labels)}")


import random
for label_file in random.sample(train_labels, min(5, len(train_labels))):
    with open(label_file) as f:
        lines = f.readlines()
    print(f"\n{label_file.name}: {len(lines)} objects")
    for line in lines[:3]:  
        print(f"  {line.strip()}")
