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
    lower = cv2.inRange(hsv, (0,30,50), (10,255,255))
    upper = cv2.inRange(hsv, (160,30,50), (180,255,255))
    mask = (lower | upper) > 0
    return mask.astype(np.uint8)