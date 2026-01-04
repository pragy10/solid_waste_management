import shutil
from pathlib import Path
from tqdm import tqdm
import random

WARP_ROOT = Path("./warp")
ZEROWASTE_ROOT = Path("./zerowaste_yolo")
OUTPUT_ROOT = Path("./swm_final")


WARP_TO_PLASTIC = {
    0: 0,  # rigid_plastic -> plastic 0
    1: 0,  # soft_plastic -> plastic 1 
    2: 0,  # plastic_bag -> plastic 1
    3: 0,  # plastic_bottle -> plastic 0
    4: 0,  # plastic_container -> plastic 0
    5: 0,  # plastic_cup -> plastic 0
    6: 0,  # plastic_cutlery -> plastic 0
    7: 0,  # plastic_straw -> plastic 0 
    11: 0, # plastic_wrapper -> plastic 1
    14: 0, # Other plastic types
}

ZEROWASTE_TO_PLASTIC = {
    0: 0,  # rigid_plastic -> plastic
    1: 0,  # soft_plastic -> plastic
    
}



def merge_and_transform():
    """Merge WaRP + ZeroWaste and transform to single-class plastic detection"""
    
    
    (OUTPUT_ROOT / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels").mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_images': 0,
        'images_with_plastic': 0,
        'images_without_plastic': 0,
        'total_plastic_boxes': 0,
        'deleted_boxes': 0
    }
    
    print("="*60)
    print("MERGING & TRANSFORMING TO SINGLE-CLASS PLASTIC DETECTION")
    print("="*60)
    
    
    print("\n[1/2] Processing WaRP dataset...")
    
    warp_splits = ['train', 'test']
    for split in warp_splits:
        images_dir = WARP_ROOT / split / "images"
        labels_dir = WARP_ROOT / split / "labels"
        
        if not images_dir.exists():
            images_dir = WARP_ROOT / "images"
            labels_dir = WARP_ROOT / "labels"
        
        if not images_dir.exists():
            print(f"WaRP {split} images not found at {images_dir}")
            continue
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"WaRP {split}"):
            
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                
                label_path = img_path.with_suffix('.txt')
                if not label_path.exists():
                    print(f"Label not found for {img_path.name}")
                    continue
            
            
            plastic_lines = []
            deleted_count = 0
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        class_id = int(parts[0])
                        
                        
                        if class_id in WARP_TO_PLASTIC:
                            
                            plastic_lines.append(f"0 {' '.join(parts[1:])}\n")
                        else:
                            deleted_count += 1
            
            
            new_img_name = f"warp_{split}_{img_path.name}"
            shutil.copy(img_path, OUTPUT_ROOT / "images" / new_img_name)
            
            
            new_label_name = f"warp_{split}_{img_path.stem}.txt"
            with open(OUTPUT_ROOT / "labels" / new_label_name, 'w') as f:
                f.writelines(plastic_lines)
            
            
            stats['total_images'] += 1
            stats['deleted_boxes'] += deleted_count
            
            if plastic_lines:
                stats['images_with_plastic'] += 1
                stats['total_plastic_boxes'] += len(plastic_lines)
            else:
                stats['images_without_plastic'] += 1
    
    
    print("\n[2/2] Processing ZeroWaste dataset...")
    
    zw_splits = ['train', 'test', 'val']
    for split in zw_splits:
        images_dir = ZEROWASTE_ROOT / split / "images"
        labels_dir = ZEROWASTE_ROOT / split / "labels"
        
        if not images_dir.exists():
            print(f"ZeroWaste {split} not found at {images_dir}")
            continue
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"ZeroWaste {split}"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                print(f"Label not found for {img_path.name}")
                continue
            
            
            plastic_lines = []
            deleted_count = 0
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    
                    
                    if class_id in ZEROWASTE_TO_PLASTIC:
                        
                        plastic_lines.append(f"0 {' '.join(parts[1:])}\n")
                    else:
                        deleted_count += 1
            
            
            new_img_name = f"zw_{split}_{img_path.name}"
            shutil.copy(img_path, OUTPUT_ROOT / "images" / new_img_name)
            
            
            new_label_name = f"zw_{split}_{img_path.stem}.txt"
            with open(OUTPUT_ROOT / "labels" / new_label_name, 'w') as f:
                f.writelines(plastic_lines)
            
            
            stats['total_images'] += 1
            stats['deleted_boxes'] += deleted_count
            
            if plastic_lines:
                stats['images_with_plastic'] += 1
                stats['total_plastic_boxes'] += len(plastic_lines)
            else:
                stats['images_without_plastic'] += 1
    
    
    print("\n" + "="*60)
    print("MERGE & TRANSFORMATION COMPLETE!")
    print("="*60)
    print(f"Total images:               {stats['total_images']}")
    print(f"Images WITH plastic:        {stats['images_with_plastic']} ({stats['images_with_plastic']/stats['total_images']*100:.1f}%)")
    print(f"Images WITHOUT plastic:     {stats['images_without_plastic']} ({stats['images_without_plastic']/stats['total_images']*100:.1f}%)")
    print(f"Total plastic boxes kept:   {stats['total_plastic_boxes']}")
    print(f"Non-plastic boxes deleted:  {stats['deleted_boxes']}")
    print(f"\nOutput location: {OUTPUT_ROOT.absolute()}")
    print("\nClass distribution:")
    print(f"  Class 0 (plastic): {stats['total_plastic_boxes']} instances")
    print(f"  Negative samples:  {stats['images_without_plastic']} images")
    
    return stats

if __name__ == "__main__":
    stats = merge_and_transform()
    
    
