"""
generate_map_svg.py
===================
Converts the Peru district shapefile to a lightweight inline SVG for the website.
Each district <path> gets data-u="UBIGEO" (the 6-char INEI shapefile code)
which the frontend uses to match against r1_districts.json / live.json.

Run ONCE before deploying.

REQUIREMENTS:
  pip install geopandas shapely

USAGE:
  python generate_map_svg.py

OUTPUT:
  data/peru_districts.svg   (~250–400 KB depending on simplification)
"""

import unicodedata
import geopandas as gpd
from pathlib import Path

DATA   = Path("data")
SHP    = DATA / "dist.shp"
OUT    = DATA / "peru_districts.svg"

# SVG canvas size in pixels
SVG_W, SVG_H = 380, 640

def strip(s):
    if not isinstance(s, str): return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    ).upper().strip()

# ── Load & simplify ───────────────────────────────────────────────────────────
print("Loading shapefile…")
gdf = gpd.read_file(SHP)
print(f"  {len(gdf)} districts loaded, CRS: {gdf.crs}")

if gdf.crs and gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

# Simplify geometry to reduce SVG size (0.02° ≈ 2 km, good enough for a web map)
gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.02, preserve_topology=True)
gdf = gdf[gdf["geometry"].notna() & ~gdf["geometry"].is_empty].copy()

# The shapefile UBIGEO is already the correct key that matches r1_districts.json
gdf["shp_ubigeo"] = gdf["UBIGEO"].astype(str).str.zfill(6)
gdf["dist_name"]  = gdf["DISTRITO"].apply(strip)
gdf["dept_name"]  = gdf["DEPARTAMEN"].apply(strip)

# ── Project to SVG space ──────────────────────────────────────────────────────
bounds = gdf.total_bounds   # (lon_min, lat_min, lon_max, lat_max)
lon_min, lat_min, lon_max, lat_max = bounds
print(f"  Bounds: lon [{lon_min:.2f}, {lon_max:.2f}]  lat [{lat_min:.2f}, {lat_max:.2f}]")

def geo_to_svg(lon, lat):
    x = (lon - lon_min) / (lon_max - lon_min) * SVG_W
    y = (lat_max - lat) / (lat_max - lat_min) * SVG_H   # flip Y
    return round(x, 1), round(y, 1)

def ring_to_d(coords):
    """Convert a list of (lon, lat) to an SVG path 'd' string."""
    pts = [geo_to_svg(lon, lat) for lon, lat in coords]
    # Deduplicate consecutive identical points
    deduped = [pts[0]]
    for pt in pts[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)
    if len(deduped) < 3:
        return ""
    d = f"M{deduped[0][0]},{deduped[0][1]}"
    for x, y in deduped[1:]:
        d += f"L{x},{y}"
    return d + "Z"

def geom_to_d(geom):
    """Convert a Shapely geometry (Polygon or MultiPolygon) to SVG 'd' string."""
    if geom is None or geom.is_empty:
        return ""
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    parts = []
    for poly in polys:
        outer = ring_to_d(list(poly.exterior.coords))
        if outer:
            parts.append(outer)
        for interior in poly.interiors:
            hole = ring_to_d(list(interior.coords))
            if hole:
                parts.append(hole)
    return " ".join(parts)

# ── Build SVG ─────────────────────────────────────────────────────────────────
print("Building SVG paths…")
path_elements = []

for _, row in gdf.iterrows():
    d = geom_to_d(row.geometry)
    if not d:
        continue
    ubigeo = str(row["shp_ubigeo"])
    dist   = str(row["dist_name"])
    dept   = str(row["dept_name"])
    # data-u carries the shapefile UBIGEO — the key the frontend uses for coloring
    path_elements.append(
        f'<path data-u="{ubigeo}" d="{d}"><title>{dist}, {dept}</title></path>'
    )

css = (
    "<style>"
    "path{fill:#d1cec6;stroke:#999;stroke-width:0.3;cursor:pointer;"
    "transition:fill 0.35s}"
    "path:hover{stroke:#222;stroke-width:0.8}"
    "</style>"
)
svg = (
    f'<svg viewBox="0 0 {SVG_W} {SVG_H}" '
    f'xmlns="http://www.w3.org/2000/svg" id="map-svg">'
    f'{css}'
    f'{"".join(path_elements)}'
    f'</svg>'
)

OUT.write_text(svg, encoding="utf-8")
size_kb = OUT.stat().st_size // 1024
print(f"✅ Saved: {OUT}")
print(f"   Size:  {size_kb} KB")
print(f"   Paths: {len(path_elements)}")
print(f"\nDone. Commit data/peru_districts.svg to your repo.")
