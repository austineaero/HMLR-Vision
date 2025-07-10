import json
import numpy as np
from skimage.draw import polygon, line
import cv2

def rasterize_json(json_path, image_shape, boundary_class=1, text_class=2):
    """Convert VIA polyline/polygon JSON to mask (0-bg,1-boundary,2-text)"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    key = [k for k in data if k.startswith("stockton_1.png")][0]
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for reg in data[key]["regions"]:
        shape = reg["shape_attributes"]
        reg_type = reg["region_attributes"].get("type")
        if shape["name"] == "polyline" and reg_type == "1":
            x, y = shape["all_points_x"], shape["all_points_y"]
            for i in range(len(x)-1):
                rr, cc = line(y[i], x[i], y[i+1], x[i+1])
                mask[rr, cc] = boundary_class
        elif shape["name"] == "polygon" and reg_type == "2":
            rr, cc = polygon(shape["all_points_y"], shape["all_points_x"], mask.shape)
            mask[rr, cc] = text_class
    return mask

def hsv_red_mask(img_rgb):
    """Returns a binary mask where red colors are detected (HSV)"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower1 = np.array([0,30,50])
    upper1 = np.array([10,255,255])
    lower2 = np.array([160,30,50])
    upper2 = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    return (mask > 0).astype(np.uint8)

# --- ADD these functions for post-processing and red-text blobs for OCR ---

def postprocess_boundary_mask(mask, min_size=100):
    """Remove small objects, close gaps in boundary mask."""
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] > min_size:
            cleaned[labels == i] = 1
    return cleaned

def extract_red_text_regions(image_rgb, min_area=30):
    """Find blobs of red text (using HSV), filter out small noise, return mask and bounding boxes."""
    red_mask = hsv_red_mask(image_rgb)
    opened = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    cleaned = np.zeros_like(opened)
    bboxes = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            cleaned[labels == i] = 1
            x, y, w, h, _ = stats[i]
            bboxes.append((x, y, w, h))
    return cleaned, bboxes