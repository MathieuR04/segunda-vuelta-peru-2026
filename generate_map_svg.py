"""
generate_map_svg.py
===================
Generates:
  data/peru_districts.svg   — district fill paths + province + dept borders
  data/centroids.json       — district centroids in SVG pixel space (for bubbles)

PROJECTION: WGS84 direct with equal aspect (lon/lat → x/y linearly).
This matches exactly what peru_maps.py produces via geopandas/matplotlib
with its default equal-aspect rendering — giving Peru the correct wide shape.

Run from project root:
  python generate_map_svg.py

REQUIREMENTS:
  pip install geopandas pandas
"""

import unicodedata, json
import geopandas as gpd
from pathlib import Path

DATA  = Path("data")
SHP   = DATA / "dist.shp"
OUT   = DATA / "peru_districts.svg"
CENT  = DATA / "centroids.json"

# SVG canvas — WGS84 equal aspect: H = W * (lat_span / lon_span)
SVG_W = 420
# Height computed from actual data bounds below

def strip(s):
    if not isinstance(s, str): return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    ).upper().strip()

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading shapefile...")
gdf = gpd.read_file(SHP)
print(f"  {len(gdf)} districts, CRS: {gdf.crs}")

# Ensure WGS84 — no reprojection, same as peru_maps.py
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)
elif gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

gdf["shp_ubigeo"] = gdf["UBIGEO"].astype(str).str.zfill(6)
gdf["dist_name"]  = gdf["DISTRITO"].apply(strip)
gdf["dept_name"]  = gdf["DEPARTAMEN"].apply(strip)
gdf["prov_name"]  = gdf["PROVINCIA"].apply(strip)

# ── Bounds & SVG height ───────────────────────────────────────────────────────
b = gdf.total_bounds   # (lon_min, lat_min, lon_max, lat_max)
lon_min, lat_min, lon_max, lat_max = b
lon_span = lon_max - lon_min
lat_span = lat_max - lat_min
SVG_H = round(SVG_W * lat_span / lon_span)
print(f"  Bounds: lon [{lon_min:.4f}, {lon_max:.4f}]  lat [{lat_min:.4f}, {lat_max:.4f}]")
print(f"  SVG canvas: {SVG_W} × {SVG_H}")

def proj(lon, lat):
    """WGS84 → SVG pixel (equal-aspect, Y-flipped)."""
    x = (lon - lon_min) / lon_span * SVG_W
    y = (lat_max - lat) / lat_span * SVG_H
    return round(x, 1), round(y, 1)

def ring_d(coords):
    pts = [proj(lon, lat) for lon, lat in coords]
    deduped = [pts[0]]
    for pt in pts[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)
    if len(deduped) < 3:
        return ""
    return "M" + "L".join(f"{x},{y}" for x, y in deduped) + "Z"

def geom_d(geom):
    if geom is None or geom.is_empty:
        return ""
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    parts = []
    for poly in polys:
        d = ring_d(list(poly.exterior.coords))
        if d: parts.append(d)
        for hole in poly.interiors:
            d = ring_d(list(hole.coords))
            if d: parts.append(d)
    return " ".join(parts)

# ── Centroids (WGS84 → SVG space) for bubble plots ───────────────────────────
print("Computing centroids...")
centroids = {}
for _, row in gdf.iterrows():
    c = row.geometry.centroid
    cx = round(float((c.x - lon_min) / lon_span * SVG_W), 1)
    cy = round(float((lat_max - c.y) / lat_span * SVG_H), 1)
    centroids[str(row["shp_ubigeo"])] = [cx, cy]

CENT.write_text(json.dumps(centroids, separators=(",", ":")))
print(f"  Saved {len(centroids)} centroids → {CENT}")

# ── Simplify ──────────────────────────────────────────────────────────────────
# In WGS84 degrees: 0.003° ≈ 300m at Peru's latitude
# Use light simplification to preserve shape quality
print("Simplifying geometries...")
gdf_s = gdf.copy()
gdf_s["geometry"] = gdf_s["geometry"].simplify(0.003, preserve_topology=True)

# Dissolve for province and dept borders BEFORE simplification gives cleaner borders
print("Dissolving province borders...")
gdf_prov = gdf.dissolve(by=["dept_name", "prov_name"]).reset_index()
gdf_prov["geometry"] = gdf_prov["geometry"].simplify(0.001, preserve_topology=True)

print("Dissolving dept borders...")
gdf_dept = gdf.dissolve(by="dept_name").reset_index()
gdf_dept["geometry"] = gdf_dept["geometry"].simplify(0.0005, preserve_topology=True)

for g in [gdf_s, gdf_prov, gdf_dept]:
    mask = g["geometry"].isna() | g["geometry"].is_empty
    g.drop(g[mask].index, inplace=True)

# ── Build SVG paths ───────────────────────────────────────────────────────────
print("Building district paths...")
dist_paths = []
for _, row in gdf_s.iterrows():
    d = geom_d(row.geometry)
    if not d: continue
    u    = str(row["shp_ubigeo"])
    name = f"{row['dist_name']}, {row['dept_name']}"
    dist_paths.append(f'<path data-u="{u}" d="{d}"><title>{name}</title></path>')

print("Building province border paths...")
prov_paths = []
for _, row in gdf_prov.iterrows():
    d = geom_d(row.geometry)
    if d: prov_paths.append(f'<path d="{d}"/>')

print("Building dept border paths...")
dept_paths = []
for _, row in gdf_dept.iterrows():
    d = geom_d(row.geometry)
    if d: dept_paths.append(f'<path d="{d}"/>')

print(f"  Districts: {len(dist_paths)}, Provinces: {len(prov_paths)}, Depts: {len(dept_paths)}")

# ── CSS ───────────────────────────────────────────────────────────────────────
# CRITICAL: district fill is NOT set here.
# JS sets it via path.style.fill = '...' which always overrides CSS fill.
# vector-effect: non-scaling-stroke keeps borders sharp at any zoom level.
css = """<style>
#districts path{fill:#d4cfc5;stroke:#c0bbb0;stroke-width:0.15;cursor:pointer;transition:fill 0.3s;vector-effect:non-scaling-stroke}
#districts path:hover{stroke:#2a2520;stroke-width:1.2;vector-effect:non-scaling-stroke}
#prov-borders path{fill:none;stroke:#666057;stroke-width:0.55;pointer-events:none;vector-effect:non-scaling-stroke}
#dept-borders path{fill:none;stroke:#1a1612;stroke-width:1.6;pointer-events:none;vector-effect:non-scaling-stroke}
</style>"""

svg = (
    f'<svg viewBox="0 0 {SVG_W} {SVG_H}" xmlns="http://www.w3.org/2000/svg" id="map-svg">\n'
    f'{css}\n'
    f'<g id="districts">{"".join(dist_paths)}</g>\n'
    f'<g id="prov-borders">{"".join(prov_paths)}</g>\n'
    f'<g id="dept-borders">{"".join(dept_paths)}</g>\n'
    f'<g id="bubbles"></g>\n'
    f'</svg>'
)

OUT.write_text(svg, encoding="utf-8")
kb = OUT.stat().st_size // 1024
print(f"\n✅  {OUT}  ({kb} KB)")
print(f"✅  {CENT}  ({CENT.stat().st_size // 1024} KB)")
print(f"\nOpen data/peru_districts.svg in a browser to verify shape.")
print("Then commit both files.")
