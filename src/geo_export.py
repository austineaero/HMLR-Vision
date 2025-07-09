import os
import cv2
import geopandas as gpd
from shapely.geometry import Polygon

def mask_to_polygons(mask, orig_w, orig_h, x_scale=1.0, y_scale=1.0, x_off=0.0, y_off=0.0):
    """Convert binary mask to polygons (in pixel coordinates)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if len(cnt) < 3: continue
        pts = cnt[:, 0, :].astype(float)
        # For this task, just use pixel coordinates (could scale to real world if you have georef)
        world = [(x_off + x_scale * x, y_off + y_scale * (orig_h - y)) for x, y in pts]
        polys.append(Polygon(world))
    return polys

def export_geopackage(boundary_mask, text_mask, orig_w, orig_h, out_path, crs="EPSG:27700", text_labels=None):
    """Export boundary and text polygons to GeoPackage."""
    records = []
    b_polys = mask_to_polygons(boundary_mask, orig_w, orig_h)
    t_polys = mask_to_polygons(text_mask, orig_w, orig_h)
    # Add boundaries
    for poly in b_polys:
        records.append({"type": "boundary", "geometry": poly})
    # Add text, optionally with label
    for i, poly in enumerate(t_polys):
        rec = {"type": "text", "geometry": poly}
        if text_labels and i < len(text_labels):
            rec["ref_num"] = text_labels[i]
        records.append(rec)
    gdf = gpd.GeoDataFrame(records, crs=crs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gdf.to_file(out_path, layer="segments", driver="GPKG")
    print(f"GeoPackage written to {out_path}")