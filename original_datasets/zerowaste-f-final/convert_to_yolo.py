import json
import os
import shutil

def create_yolo_dataset(input_base_dir, output_base_dir='yolo_dataset'):
    """
    Convert COCO format to YOLO format for all splits
    """
    
    class_names = ['rigid_plastic', 'soft_plastic', 'cardboard', 'metal']
    
    category_to_yolo = {
        1: 0,  # rigid_plastic
        4: 1,  # soft_plastic
        2: 2,  # cardboard
        3: 3   # metal
    }
    
    print("="*70)
    print("COCO to YOLO Converter")
    print("="*70)
    print(f"\nClass mapping (COCO ID -> YOLO ID):")
    print(f"  1: rigid_plastic -> 0")
    print(f"  4: soft_plastic -> 1")
    print(f"  2: cardboard -> 2")
    print(f"  3: metal -> 3\n")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split...")
        print('='*70)
        
        # Paths for this split
        json_path = os.path.join(input_base_dir, split, 'labels.json')
        images_src_dir = os.path.join(input_base_dir, split, 'data')
        
        # Check if files exist
        if not os.path.exists(json_path):
            print(f"⚠ Skipping: {json_path} not found")
            continue
        
        # Create output directories
        images_dest_dir = os.path.join(output_base_dir, split, 'images')
        labels_dest_dir = os.path.join(output_base_dir, split, 'labels')
        os.makedirs(images_dest_dir, exist_ok=True)
        os.makedirs(labels_dest_dir, exist_ok=True)
        
        # Load JSON
        print(f"Loading {json_path}...")
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        
        image_info = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        print(f"Converting {len(annotations_by_image)} images...")
        converted_images = 0
        total_annotations = 0
        
        for img_id, annotations in annotations_by_image.items():
            img = image_info[img_id]
            img_filename = img['file_name']
            img_width = img['width']
            img_height = img['height']
            
            # Copy image
            src_image = os.path.join(images_src_dir, img_filename)
            dest_image = os.path.join(images_dest_dir, img_filename)
            
            if os.path.exists(src_image):
                shutil.copy2(src_image, dest_image)
            else:
                print(f"  ⚠ Image not found: {img_filename}")
                continue
            
            # Create label file
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            label_path = os.path.join(labels_dest_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    coco_cat_id = ann['category_id']
                    class_id = category_to_yolo.get(coco_cat_id)
                    
                    if class_id is None:
                        continue
                    
                    segmentation = ann['segmentation'][0] if ann['segmentation'] else []
                    if not segmentation:
                        continue
                    
                    # Normalize coordinates
                    normalized_coords = []
                    for i in range(0, len(segmentation), 2):
                        x = max(0.0, min(1.0, segmentation[i] / img_width))
                        y = max(0.0, min(1.0, segmentation[i + 1] / img_height))
                        normalized_coords.extend([x, y])
                    
                    line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords])
                    f.write(line + '\n')
                    total_annotations += 1
            
            converted_images += 1
            if converted_images % 100 == 0:
                print(f"  Processed {converted_images} images...")
        
        print(f"\n✓ {split.upper()} complete:")
        print(f"  - Images: {converted_images}")
        print(f"  - Annotations: {total_annotations}")
    
    # Create data.yaml
    yaml_path = os.path.join(output_base_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write("# YOLO Dataset Configuration\n\n")
        f.write(f"path: {os.path.abspath(output_base_dir)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write("# Classes\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for idx, class_name in enumerate(class_names):
            f.write(f"  {idx}: {class_name}\n")
    
    # Create classes.txt
    classes_path = os.path.join(output_base_dir, 'classes.txt')
    with open(classes_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"\n{'='*70}")
    print("✓ CONVERSION COMPLETE!")
    print('='*70)
    print(f"Output: {os.path.abspath(output_base_dir)}")
    print("\nFiles created:")
    print("  - data.yaml")
    print("  - classes.txt")
    print("  - train/images/ and train/labels/")
    print("  - val/images/ and val/labels/")
    print("  - test/images/ and test/labels/")
    print('='*70)


# Run the conversion
# Current directory is D:\swm\original_datasets\zerowaste-f-final
# So we use '.' for current directory
create_yolo_dataset('.', 'yolo_dataset')
