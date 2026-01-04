import os
import shutil
from pathlib import Path

corrupted = Path('/kaggle/working/plastic_large')
if corrupted.exists():
    shutil.rmtree(corrupted)
    print("Deleted corrupted checkpoint")

import torch
import gc

torch.cuda.empty_cache()
gc.collect()

os.system('pip install -q ultralytics opencv-python')

from ultralytics import YOLO
import yaml

OUTPUT_DIR = '/kaggle/working'
DATA_PATH = '/kaggle/input/swm-final-split/swm_final_split'

with open(f'{OUTPUT_DIR}/data.yaml', 'w') as f:
    yaml.dump({
        'path': DATA_PATH,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': {0: 'plastic'}
    }, f)

model = YOLO('yolov8l.pt')

torch.cuda.empty_cache()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Starting FRESH training (no resume)")

model.train(
    data=f'{OUTPUT_DIR}/data.yaml',
    epochs=100,
    batch=6,
    imgsz=640,
    patience=25,
    device=0,
    workers=2,
    project=OUTPUT_DIR,
    name='plastic_large_v2',
    resume=False,
    save_period=5,
    exist_ok=False,
    amp=True,
    cache=False,
    verbose=True
)

best_model = YOLO(f"{OUTPUT_DIR}/plastic_large_v2/weights/best.pt")
metrics = best_model.val(data=f'{OUTPUT_DIR}/data.yaml', split='test')

print(f"\n{'='*60}")
print(f" TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"mAP@0.5: {metrics.box.map50:.3f} ({metrics.box.map50*100:.1f}%)")
print(f"mAP@0.5-95: {metrics.box.map:.3f} ({metrics.box.map*100:.1f}%)")
print(f"Precision: {metrics.box.mp:.3f} ({metrics.box.mp*100:.1f}%)")
print(f"Recall: {metrics.box.mr:.3f} ({metrics.box.mr*100:.1f}%)")
print(f"{'='*60}\n")

weights = Path(OUTPUT_DIR) / 'plastic_large_v2' / 'weights'
if (weights / 'best.pt').exists():
    shutil.copy(weights / 'best.pt', f'{OUTPUT_DIR}/DOWNLOAD_LARGE_best.pt')
    print(" Saved: DOWNLOAD_LARGE_best.pt")
if (weights / 'last.pt').exists():
    shutil.copy(weights / 'last.pt', f'{OUTPUT_DIR}/DOWNLOAD_LARGE_last.pt')
    print(" Saved: DOWNLOAD_LARGE_last.pt")

