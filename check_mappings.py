import cv2
import json
import random
from pathlib import Path


TARGET_NAMES = {
    0: 'RIGID_PLASTIC', 1: 'SOFT_PLASTIC', 2: 'GLASS', 
    3: 'METAL', 4: 'CARDBOARD', 5: 'PAPER'
}


PATH_WARP = Path("./warp")
PATH_TRASHNET = Path("./trashnet")
PATH_ZEROWASTE_ROOT = Path("./zerowaste/train") 


WARP_MAP = {
    0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:3, 9:4, 10:4, 
    11:0, 12:0, 13:4, 14:0, 25:2, 26:2, 27:2
}

ZW_MAP = {
    1: 0, # rigid_plastic -> rigid
    2: 4, # cardboard -> cardboard
    3: 3, # metal -> metal
    4: 1  # soft_plastic -> soft
}

TN_MAP = {0: 4, 1: 2, 2: 3, 3: 5, 4: 0, 5: -1}


def draw(img, cls_id, bbox, fmt='yolo'):
    h, w = img.shape[:2]
    if fmt == 'yolo': # x_c, y_c, w, h
        x = int((bbox[0] - bbox[2]/2) * w)
        y = int((bbox[1] - bbox[3]/2) * h)
        bw = int(bbox[2] * w)
        bh = int(bbox[3] * h)
    else: 
        x, y, bw, bh = [int(v) for v in bbox]
        
    cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
    label = TARGET_NAMES.get(cls_id, f"ID {cls_id}")
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img


def check_zerowaste():
    print("\n--- Checking ZeroWaste ---")
    json_file = PATH_ZEROWASTE_ROOT / "labels.json"
    
    if not json_file.exists():
        print(f"labels.json not found at {json_file}")
        return

    with open(json_file) as f: data = json.load(f)
    
    images = random.sample(data['images'], 15)
    
    for img_info in images:
        fname = img_info['file_name']
        img_path = PATH_ZEROWASTE_ROOT / "data" / fname
        if not img_path.exists():
             img_path = PATH_ZEROWASTE_ROOT / fname
        
        if not img_path.exists():
            print(f"Image not found: {fname}")
            continue
            
        img = cv2.imread(str(img_path))
        img_id = img_info['id']
        anns = [a for a in data['annotations'] if a['image_id'] == img_id]
        
        found_box = False
        for ann in anns:
            cid = ann['category_id']
            if cid in ZW_MAP:
                new_id = ZW_MAP[cid]
                img = draw(img, new_id, ann['bbox'], fmt='coco')
                found_box = True
        
        if found_box:
            cv2.imshow(f"ZW: {fname}", cv2.resize(img, (800, 600)))
            if cv2.waitKey(0) == 27: return # ESC to quit
    cv2.destroyAllWindows()

def check_warp():
    print("\n--- Checking WaRP ---")
    images = list(PATH_WARP.rglob("*.jpg"))[:15]
    for p in images:
        img = cv2.imread(str(p))
        txt = p.with_suffix(".txt")
        if not txt.exists(): 
            if 'images' in p.parts:
                parts = list(p.parts)
                parts[parts.index('images')] = 'labels'
                txt = Path(*parts).with_suffix(".txt")
        
        if txt.exists():
            with open(txt) as f:
                for line in f:
                    parts = line.split()
                    cid = int(parts[0])
                    if cid in WARP_MAP:
                        img = draw(img, WARP_MAP[cid], [float(x) for x in parts[1:]])
            cv2.imshow("WaRP", cv2.resize(img, (800, 600)))
            if cv2.waitKey(0) == 27: return
    cv2.destroyAllWindows()

def check_trashnet():
    print("\n--- Checking TrashNet ---")
    images = list(PATH_TRASHNET.rglob("*.jpg"))[:15]
    for p in images:
        img = cv2.imread(str(p))
        txt = p.with_suffix(".txt")
        if not txt.exists() and 'images' in p.parts:
             parts = list(p.parts)
             parts[parts.index('images')] = 'labels'
             txt = Path(*parts).with_suffix(".txt")

        if txt and txt.exists():
            with open(txt) as f:
                for line in f:
                    if line.startswith('#'): continue
                    parts = line.split()
                    cid = int(parts[0])
                    if cid in TN_MAP:
                        img = draw(img, TN_MAP[cid], [float(x) for x in parts[1:]])
            cv2.imshow("TrashNet", cv2.resize(img, (600, 600)))
            if cv2.waitKey(0) == 27: return
    cv2.destroyAllWindows()



check_warp()
check_trashnet()
check_zerowaste()
