import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ===== CONFIGURATION =====
SOURCE_ROOT = Path("./swm_final")
OUTPUT_ROOT = Path("./swm_final_split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# =========================

def split_dataset():
    """Split merged dataset into train/val/test"""
    
    print("="*60)
    print("SPLITTING DATASET INTO TRAIN/VAL/TEST")
    print("="*60)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (OUTPUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Get all images
    images_dir = SOURCE_ROOT / "images"
    labels_dir = SOURCE_ROOT / "labels"
    
    all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    print(f"\nTotal images: {len(all_images)}")
    
    # Shuffle
    random.shuffle(all_images)
    
    # Calculate split points
    total = len(all_images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    splits = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }
    
    print(f"Train: {len(splits['train'])} ({TRAIN_RATIO*100:.0f}%)")
    print(f"Val:   {len(splits['val'])} ({VAL_RATIO*100:.0f}%)")
    print(f"Test:  {len(splits['test'])} ({TEST_RATIO*100:.0f}%)")
    
    # Copy files
    for split_name, img_list in splits.items():
        print(f"\nCopying {split_name} split...")
        
        for img_path in tqdm(img_list):
            # Copy image
            shutil.copy(
                img_path,
                OUTPUT_ROOT / split_name / "images" / img_path.name
            )
            
            # Copy label
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(
                    label_path,
                    OUTPUT_ROOT / split_name / "labels" / f"{img_path.stem}.txt"
                )
    
    # Create data.yaml
    yaml_content = f"""# SWM Plastic Detection Dataset
path: {OUTPUT_ROOT.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: 1
names: ['plastic']

# Augmentation (YOLOv8 built-in)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 15
translate: 0.1
scale: 0.5
fliplr: 0.5
mosaic: 1.0
"""
    
    with open(OUTPUT_ROOT / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print("\n" + "="*60)
    print("✅ DATASET SPLIT COMPLETE!")
    print("="*60)
    print(f"Output: {OUTPUT_ROOT.absolute()}")
    print(f"\nNext: Train with:")
    print(f"yolo detect train data={OUTPUT_ROOT.absolute()}/data.yaml model=yolov8n.pt epochs=100")

if __name__ == "__main__":
    split_dataset()
