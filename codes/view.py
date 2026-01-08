
"""
YOLO Dataset Visualization Tool
Visualizes images with bounding boxes to verify class mappings
Perfect for checking if remapping was done correctly
"""

import os
import cv2
import random
from pathlib import Path

def visualize_yolo_dataset(dataset_path, num_samples=20, split='train'):
    """
    Visualize random samples from YOLO dataset with bounding boxes

    Args:
        dataset_path: Path to your dataset (e.g., 'merged_dataset' or 'warp_remapped')
        num_samples: Number of images to display
        split: Which split to visualize ('train', 'val', 'test')
    """

    
    class_names = ['rigid_plastic', 'soft_plastic', 'cardboard', 'metal']
    class_colors = {
        0: (255, 0, 0),      
        1: (0, 255, 0),      
        2: (0, 255, 255),    
        3: (255, 0, 255)     
    }

    
    images_dir = os.path.join(dataset_path, split, 'images')
    labels_dir = os.path.join(dataset_path, split, 'labels')

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return

    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found: {labels_dir}")
        return

    
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(image_files) == 0:
        print(f"No images found in {images_dir}")
        return

    
    num_samples = min(num_samples, len(image_files))
    selected_images = random.sample(image_files, num_samples)

    print("="*70)
    print(f"YOLO Dataset Visualizer - {split.upper()} Split")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Total images: {len(image_files)}")
    print(f"Showing: {num_samples} random samples")
    print(f"\nClasses:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    print("\nPress any key to see next image, 'q' to quit")
    print("="*70 + "\n")

    for img_file in selected_images:
        
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load: {img_file}")
            continue

        h, w = img.shape[:2]

        
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        annotations_count = {i: 0 for i in range(len(class_names))}

        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                annotations_count[class_id] += 1

                
                if len(parts) == 5:
                    x_center, y_center, width, height = map(float, parts[1:5])

                    
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)

                    
                    color = class_colors.get(class_id, (128, 128, 128))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    
                    label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(img, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                
                elif len(parts) > 5:
                    coords = list(map(float, parts[1:]))
                    points = []

                    for i in range(0, len(coords), 2):
                        if i+1 < len(coords):
                            x = int(coords[i] * w)
                            y = int(coords[i+1] * h)
                            points.append([x, y])

                    if len(points) >= 3:
                        import numpy as np
                        points = np.array(points, dtype=np.int32)
                        color = class_colors.get(class_id, (128, 128, 128))

                        
                        cv2.polylines(img, [points], True, color, 2)

                        
                        overlay = img.copy()
                        cv2.fillPoly(overlay, [points], color)
                        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

                        
                        M = cv2.moments(points)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                            cv2.putText(img, label, (cx-30, cy), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            print(f"No label file for: {img_file}")

        
        info_text = f"{img_file} | "
        for class_id, count in annotations_count.items():
            if count > 0:
                info_text += f"{class_names[class_id]}: {count}  "

        cv2.putText(img, info_text, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(img, info_text, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        
        cv2.imshow('YOLO Dataset Viewer', img)

        
        print(f"Image: {img_file}")
        print(f"  Size: {w}x{h}")
        print(f"  Annotations: ", end="")
        for class_id, count in annotations_count.items():
            if count > 0:
                print(f"{class_names[class_id]}={count} ", end="")
        print("\n")

        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:  
            print("\nVisualization stopped by user")
            break

    cv2.destroyAllWindows()
    print("\nVisualization complete!")


if __name__ == "__main__":
    import sys

    print("\n" + "="*70)
    print("YOLO Dataset Visualization Tool")
    print("="*70 + "\n")

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        split = sys.argv[2] if len(sys.argv) > 2 else 'train'
        num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    else:
        print("Visualize YOLO annotations with bounding boxes/segmentation masks")
        print("\nExamples:")
        print("  Merged dataset: merged_dataset")
        print("  WaRP remapped: warp_remapped")
        print("  Zerowaste: yolo_dataset")
        print()

        dataset_path = input("Dataset path: ").strip().strip('\"').strip("'")

        split = input("Split (train/val/test) [train]: ").strip().lower()
        if not split:
            split = 'train'

        num_str = input("Number of samples to show [20]: ").strip()
        num_samples = int(num_str) if num_str else 20

    if not os.path.exists(dataset_path):
        print(f"\nError: Dataset path not found: {dataset_path}")
        sys.exit(1)

    try:
        visualize_yolo_dataset(dataset_path, num_samples, split)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)