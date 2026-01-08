#!/usr/bin/env python3
"""
WaRP Dataset Remapper: 28 Classes -> 4 Classes
Remaps WaRP dataset to match your 4-class scheme:
0: rigid_plastic
1: soft_plastic
2: cardboard
3: metal
"""

import os
import shutil

def remap_warp_to_4_classes(input_base_dir, output_base_dir='warp_remapped'):
    """
    Remap WaRP dataset (28 classes) to 4 classes
    """

    # Target classes in your specified order
    target_classes = ['rigid_plastic', 'soft_plastic', 'cardboard', 'metal']

    # Mapping: WaRP class ID (0-27) -> Target class ID (0-3)
    # Based on WaRP class names from classes.txt:
    warp_to_target = {
        # Rigid plastic bottles -> rigid_plastic (0)
        0: 0,   # bottle-blue
        1: 0,   # bottle-green
        2: 0,   # bottle-dark
        3: 0,   # bottle-milk
        4: 0,   # bottle-transp
        5: 0,   # bottle-multicolor
        6: 0,   # bottle-yogurt
        7: 0,   # bottle-oil
        15: 0,  # bottle-blue-full
        16: 0,  # bottle-transp-full
        17: 0,  # bottle-dark-full
        18: 0,  # bottle-green-full
        19: 0,  # bottle-multicolor-full
        20: 0,  # bottle-milk-full
        21: 0,  # bottle-oil-full
        23: 0,  # bottle-blue5l
        24: 0,  # bottle-blue5l-full
        25: 0,  # glass-transp (mapped as rigid plastic)
        26: 0,  # glass-dark (mapped as rigid plastic)
        27: 0,  # glass-green (mapped as rigid plastic)

        # Detergent containers -> soft_plastic (1)
        11: 1,  # detergent-color
        12: 1,  # detergent-transparent
        13: 1,  # detergent-box
        22: 1,  # detergent-white

        # Cardboard containers -> cardboard (2)
        9: 2,   # juice-cardboard
        10: 2,  # milk-cardboard

        # Metal cans -> metal (3)
        8: 3,   # cans
        14: 3,  # canister
    }

    warp_class_names = [
        'bottle-blue', 'bottle-green', 'bottle-dark', 'bottle-milk',
        'bottle-transp', 'bottle-multicolor', 'bottle-yogurt', 'bottle-oil',
        'cans', 'juice-cardboard', 'milk-cardboard', 'detergent-color',
        'detergent-transparent', 'detergent-box', 'canister',
        'bottle-blue-full', 'bottle-transp-full', 'bottle-dark-full',
        'bottle-green-full', 'bottle-multicolor-full', 'bottle-milk-full',
        'bottle-oil-full', 'detergent-white', 'bottle-blue5l',
        'bottle-blue5l-full', 'glass-transp', 'glass-dark', 'glass-green'
    ]

    print("="*70)
    print("WaRP Dataset Remapper: 28 Classes -> 4 Classes")
    print("="*70)
    print("\nMapping scheme:\n")

    # Display mapping by target class
    for target_id, target_name in enumerate(target_classes):
        print(f"{target_id}: {target_name.upper()}")
        mapped_classes = [warp_class_names[old_id] for old_id, new_id in warp_to_target.items() if new_id == target_id]
        for cls in mapped_classes:
            print(f"   - {cls}")
        print()

    splits = ['train', 'test']
    total_files_processed = 0
    total_annotations_remapped = 0

    for split in splits:
        print(f"{'='*70}")
        print(f"Processing {split.upper()} split...")
        print('='*70)

        # Paths
        labels_src_dir = os.path.join(input_base_dir, split, 'labels')
        images_src_dir = os.path.join(input_base_dir, split, 'images')

        labels_dest_dir = os.path.join(output_base_dir, split, 'labels')
        images_dest_dir = os.path.join(output_base_dir, split, 'images')

        # Check if source exists
        if not os.path.exists(labels_src_dir):
            print(f"⚠ Warning: {labels_src_dir} not found. Skipping {split}.")
            continue

        # Create output directories
        os.makedirs(labels_dest_dir, exist_ok=True)
        os.makedirs(images_dest_dir, exist_ok=True)

        # Get all label files
        label_files = [f for f in os.listdir(labels_src_dir) if f.endswith('.txt')]
        print(f"Found {len(label_files)} label files\n")

        files_processed = 0
        annotations_in_split = 0

        for label_file in label_files:
            src_label_path = os.path.join(labels_src_dir, label_file)
            dest_label_path = os.path.join(labels_dest_dir, label_file)

            # Read and remap labels
            with open(src_label_path, 'r') as f:
                lines = f.readlines()

            remapped_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:  # At least class_id + 4 coordinates
                    continue

                old_class_id = int(parts[0])

                # Remap class ID
                new_class_id = warp_to_target.get(old_class_id, 0)  # Default to rigid_plastic

                # Keep all coordinates unchanged
                coords = ' '.join(parts[1:])
                remapped_lines.append(f"{new_class_id} {coords}\n")
                annotations_in_split += 1

            # Write remapped labels
            with open(dest_label_path, 'w') as f:
                f.writelines(remapped_lines)

            # Copy corresponding image
            image_file = label_file.replace('.txt', '.jpg')

            # Try different image extensions
            if not os.path.exists(os.path.join(images_src_dir, image_file)):
                for ext in ['.png', '.PNG', '.JPG', '.jpeg', '.JPEG']:
                    alt_image = label_file.replace('.txt', ext)
                    if os.path.exists(os.path.join(images_src_dir, alt_image)):
                        image_file = alt_image
                        break

            src_image_path = os.path.join(images_src_dir, image_file)
            dest_image_path = os.path.join(images_dest_dir, image_file)

            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, dest_image_path)
            else:
                print(f"  ⚠ Image not found: {image_file}")

            files_processed += 1
            if files_processed % 500 == 0:
                print(f"  Processed {files_processed} files...")

        total_files_processed += files_processed
        total_annotations_remapped += annotations_in_split

        print(f"\n✓ {split.upper()} complete:")
        print(f"  - Files: {files_processed}")
        print(f"  - Annotations remapped: {annotations_in_split}\n")

    # Create new data.yaml
    yaml_path = os.path.join(output_base_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write("# WaRP Dataset Remapped to 4 Classes\n")
        f.write("# Original: 28 classes -> New: 4 classes\n")
        f.write("# Classes: rigid_plastic, soft_plastic, cardboard, metal\n\n")
        f.write(f"path: {os.path.abspath(output_base_dir)}\n")
        f.write("train: train/images\n")
        f.write("test: test/images\n\n")
        f.write("# Classes\n")
        f.write(f"nc: {len(target_classes)}\n")
        f.write("names:\n")
        for idx, class_name in enumerate(target_classes):
            f.write(f"  {idx}: {class_name}\n")

    # Create classes.txt
    classes_path = os.path.join(output_base_dir, 'classes.txt')
    with open(classes_path, 'w') as f:
        for class_name in target_classes:
            f.write(f"{class_name}\n")

    # Create detailed mapping reference
    mapping_path = os.path.join(output_base_dir, 'class_mapping_reference.txt')
    with open(mapping_path, 'w') as f:
        f.write("WaRP Dataset Class Remapping Reference\n")
        f.write("="*70 + "\n\n")
        f.write("Original WaRP (28 classes) -> New (4 classes)\n\n")

        for target_id, target_name in enumerate(target_classes):
            f.write(f"\n{target_id}: {target_name.upper()}\n")
            f.write("-" * 40 + "\n")
            for old_id, new_id in sorted(warp_to_target.items()):
                if new_id == target_id:
                    f.write(f"  WaRP class {old_id:2d}: {warp_class_names[old_id]}\n")

    print(f"{'='*70}")
    print("✓ REMAPPING COMPLETE!")
    print('='*70)
    print(f"\nOutput directory: {os.path.abspath(output_base_dir)}")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total annotations remapped: {total_annotations_remapped}")
    print("\nGenerated files:")
    print("  ✓ data.yaml (4-class configuration)")
    print("  ✓ classes.txt (4 classes in order)")
    print("  ✓ class_mapping_reference.txt (detailed mapping)")
    print("  ✓ train/images/ and train/labels/")
    print("  ✓ test/images/ and test/labels/")
    print('='*70)
    print("\nYour 4 classes are:")
    for idx, cls in enumerate(target_classes):
        print(f"  {idx}: {cls}")
    print()

    return output_base_dir


if __name__ == "__main__":
    import sys

    print("\n" + "="*70)
    print("WaRP Dataset Remapper")
    print("="*70 + "\n")

    # Get input path
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        print("Enter the path to your WaRP dataset folder:")
        print("(The folder containing train/ and test/ subdirectories)")
        print("\nExample: D:\\swm\\original_datasets\\warp\n")
        input_dir = input("Path: ").strip()
        input_dir = input_dir.strip('\"').strip("'")

    # Get output path
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        print("\nEnter output folder name (default: warp_remapped):")
        output_dir = input("Output folder: ").strip()
        if not output_dir:
            output_dir = 'warp_remapped'

    # Validate
    if not os.path.exists(input_dir):
        print(f"\n❌ Error: Directory not found: {input_dir}")
        sys.exit(1)

    # Run remapping
    try:
        remap_warp_to_4_classes(input_dir, output_dir)
        print("\n✓ Success! Your remapped WaRP dataset is ready.\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)