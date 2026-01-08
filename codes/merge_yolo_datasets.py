#!/usr/bin/env python3
"""
YOLO Dataset Merger
Merges multiple YOLO format datasets with the same classes
Combines: zerowaste-f-final (converted) + WaRP (remapped)
Target classes: rigid_plastic, soft_plastic, cardboard, metal
"""

import os
import shutil
from collections import defaultdict

def merge_yolo_datasets(dataset_paths, output_dir='merged_dataset', dataset_names=None):
    """
    Merge multiple YOLO datasets with the same class structure

    Args:
        dataset_paths: List of paths to datasets to merge
        output_dir: Output directory for merged dataset
        dataset_names: Optional list of names for each dataset (for prefixing)
    """

    target_classes = ['rigid_plastic', 'soft_plastic', 'cardboard', 'metal']

    if dataset_names is None:
        dataset_names = [f"dataset{i+1}" for i in range(len(dataset_paths))]

    print("="*70)
    print("YOLO Dataset Merger")
    print("="*70)
    print(f"\nMerging {len(dataset_paths)} datasets:")
    for i, (path, name) in enumerate(zip(dataset_paths, dataset_names)):
        print(f"  {i+1}. {name}: {path}")
    print(f"\nTarget classes: {', '.join(target_classes)}")
    print(f"Output: {output_dir}\n")

    # Process each split
    splits = ['train', 'val', 'test']

    stats = defaultdict(lambda: {'images': 0, 'annotations': 0})
    total_stats = {'images': 0, 'annotations': 0}

    for split in splits:
        print(f"{'='*70}")
        print(f"Processing {split.upper()} split...")
        print('='*70)

        # Create output directories
        output_images_dir = os.path.join(output_dir, split, 'images')
        output_labels_dir = os.path.join(output_dir, split, 'labels')
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        split_images = 0
        split_annotations = 0

        # Process each dataset
        for dataset_path, dataset_name in zip(dataset_paths, dataset_names):
            print(f"\n  Processing {dataset_name}...")

            # Check for both possible directory structures
            images_src_dirs = [
                os.path.join(dataset_path, split, 'images'),
                os.path.join(dataset_path, split, 'data')  # For zerowaste structure
            ]
            labels_src_dirs = [
                os.path.join(dataset_path, split, 'labels'),
                os.path.join(dataset_path, split, 'labels')
            ]

            images_src_dir = None
            labels_src_dir = None

            for img_dir, lbl_dir in zip(images_src_dirs, labels_src_dirs):
                if os.path.exists(img_dir) and os.path.exists(lbl_dir):
                    images_src_dir = img_dir
                    labels_src_dir = lbl_dir
                    break

            if images_src_dir is None or labels_src_dir is None:
                print(f"    ⚠ {split} split not found in {dataset_name}, skipping")
                continue

            # Get all label files
            if not os.path.exists(labels_src_dir):
                print(f"    ⚠ Labels directory not found, skipping")
                continue

            label_files = [f for f in os.listdir(labels_src_dir) if f.endswith('.txt')]

            if len(label_files) == 0:
                print(f"    ⚠ No label files found, skipping")
                continue

            dataset_images = 0
            dataset_annotations = 0

            for label_file in label_files:
                # Create unique filename by prefixing with dataset name
                base_name = os.path.splitext(label_file)[0]
                new_label_file = f"{dataset_name}_{label_file}"

                # Copy label file
                src_label = os.path.join(labels_src_dir, label_file)
                dst_label = os.path.join(output_labels_dir, new_label_file)

                # Count annotations while copying
                with open(src_label, 'r') as f:
                    label_lines = [line for line in f if line.strip()]
                    dataset_annotations += len(label_lines)

                shutil.copy2(src_label, dst_label)

                # Find and copy corresponding image
                image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                image_copied = False

                for ext in image_extensions:
                    image_file = base_name + ext
                    src_image = os.path.join(images_src_dir, image_file)

                    if os.path.exists(src_image):
                        new_image_file = f"{dataset_name}_{image_file}"
                        dst_image = os.path.join(output_images_dir, new_image_file)
                        shutil.copy2(src_image, dst_image)
                        dataset_images += 1
                        image_copied = True
                        break

                if not image_copied:
                    print(f"    ⚠ Image not found for {label_file}")

            print(f"    ✓ Added {dataset_images} images, {dataset_annotations} annotations")

            split_images += dataset_images
            split_annotations += dataset_annotations
            stats[dataset_name]['images'] += dataset_images
            stats[dataset_name]['annotations'] += dataset_annotations

        total_stats['images'] += split_images
        total_stats['annotations'] += split_annotations

        print(f"\n  {split.upper()} totals: {split_images} images, {split_annotations} annotations")

    # Create data.yaml
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write("# Merged YOLO Dataset\n")
        f.write(f"# Combined from: {', '.join(dataset_names)}\n")
        f.write(f"# Total images: {total_stats['images']}\n")
        f.write(f"# Total annotations: {total_stats['annotations']}\n\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write("# Classes\n")
        f.write(f"nc: {len(target_classes)}\n")
        f.write("names:\n")
        for idx, class_name in enumerate(target_classes):
            f.write(f"  {idx}: {class_name}\n")

    # Create classes.txt
    classes_path = os.path.join(output_dir, 'classes.txt')
    with open(classes_path, 'w') as f:
        for class_name in target_classes:
            f.write(f"{class_name}\n")

    # Create merge statistics file
    stats_path = os.path.join(output_dir, 'merge_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("Merged Dataset Statistics\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Images: {total_stats['images']}\n")
        f.write(f"Total Annotations: {total_stats['annotations']}\n\n")
        f.write("Breakdown by source dataset:\n")
        f.write("-"*70 + "\n")
        for dataset_name in dataset_names:
            f.write(f"\n{dataset_name}:\n")
            f.write(f"  Images: {stats[dataset_name]['images']}\n")
            f.write(f"  Annotations: {stats[dataset_name]['annotations']}\n")

    print(f"\n{'='*70}")
    print("✓ MERGE COMPLETE!")
    print('='*70)
    print(f"\nOutput directory: {os.path.abspath(output_dir)}")
    print(f"\nTotal statistics:")
    print(f"  Images: {total_stats['images']}")
    print(f"  Annotations: {total_stats['annotations']}")
    print(f"\nBreakdown by dataset:")
    for dataset_name in dataset_names:
        print(f"  {dataset_name}:")
        print(f"    - Images: {stats[dataset_name]['images']}")
        print(f"    - Annotations: {stats[dataset_name]['annotations']}")
    print(f"\nGenerated files:")
    print("  ✓ data.yaml")
    print("  ✓ classes.txt")
    print("  ✓ merge_statistics.txt")
    print("  ✓ train/images/ and train/labels/")
    print("  ✓ val/images/ and val/labels/")
    print("  ✓ test/images/ and test/labels/")
    print('='*70)
    print("\nNote: Image files are prefixed with dataset name to avoid conflicts")
    print()

    return output_dir


if __name__ == "__main__":
    import sys

    print("\n" + "="*70)
    print("YOLO Dataset Merger")
    print("="*70 + "\n")

    # Example usage
    print("This script merges multiple YOLO datasets with the same classes.")
    print("\nDefault configuration:")
    print("  Dataset 1: yolo_dataset (zerowaste-f-final converted)")
    print("  Dataset 2: warp_remapped (WaRP remapped)")
    print("  Output: merged_dataset")
    print()

    if len(sys.argv) >= 3:
        # Command line: dataset1 dataset2 [output]
        dataset1 = sys.argv[1]
        dataset2 = sys.argv[2]
        output = sys.argv[3] if len(sys.argv) > 3 else 'merged_dataset'
    else:
        # Interactive mode
        print("Enter dataset paths:")
        dataset1 = input("Dataset 1 path (zerowaste): ").strip().strip('\"').strip("'")
        dataset2 = input("Dataset 2 path (warp): ").strip().strip('\"').strip("'")

        output = input("\nOutput directory (default: merged_dataset): ").strip()
        if not output:
            output = 'merged_dataset'

    # Validate paths
    if not os.path.exists(dataset1):
        print(f"\n❌ Error: Dataset 1 not found: {dataset1}")
        sys.exit(1)

    if not os.path.exists(dataset2):
        print(f"\n❌ Error: Dataset 2 not found: {dataset2}")
        sys.exit(1)

    # Run merge
    try:
        dataset_paths = [dataset1, dataset2]
        dataset_names = ['zerowaste', 'warp']

        merge_yolo_datasets(dataset_paths, output, dataset_names)
        print("\n✓ Success! Your merged dataset is ready.\n")
    except Exception as e:
        print(f"\n❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
