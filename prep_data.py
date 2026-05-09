"""
prep_data.py
============
Run ONCE (now, before election day) to bake the two static reference datasets.

Outputs:
  data/r1_districts.json    — 2026 first-round results keyed by shapefile UBIGEO
  data/r2021_districts.json — 2021 runoff results keyed by shapefile UBIGEO

KEY UBIGEO NOTE
---------------
mesas_metadata uses ONPE's internal id_ubigeo (e.g. 10101).
dist.shp uses INEI's UBIGEO string (e.g. "010101").
These are NOT the same coding system — only ~53% overlap directly.
ubigeo_lookup.csv bridges them by matching on (region, provincia, distrito) names.

The join chain is always:
  mesas_metadata.id_ubigeo
    → ubigeo_lookup.id_ubigeo        (ONPE side)
    → ubigeo_lookup.(region/prov/dist) names
    → dist.shp.(DEPARTAMEN/PROVINCIA/DISTRITO) names
    → dist.shp.UBIGEO                (shapefile side — used as the map key)

All records in both JSONs are keyed by the shapefile UBIGEO ("u" field)
so the frontend can directly paint SVG paths (which carry data-u="UBIGEO").

REQUIREMENTS:
  pip install pandas pyarrow geopandas

USAGE:
  python prep_data.py
"""

import unicodedata
import pandas as pd
import numpy as np
import json
import geopandas as gpd
from pathlib import Path

DATA = Path("data")

KEIKO   = "FUERZA POPULAR"
SANCHEZ = "JUNTOS POR EL PERÚ"
NIETO   = "PARTIDO DEL BUEN GOBIERNO"
RLA     = "RENOVACIÓN POPULAR"

def strip(s):
    """Normalize string: strip accents, uppercase, trim — for name matching."""
    if not isinstance(s, str): return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    ).upper().strip()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Build the ubigeo bridge
#
# Joins ubigeo_lookup (id_ubigeo = ONPE) to dist.shp (UBIGEO = INEI/shapefile)
# via normalized (dept, prov, dist) name strings.
# Result: id_ubigeo → shp_ubigeo (the correct map key)
# ─────────────────────────────────────────────────────────────────────────────
print("Building ubigeo bridge…")

lu = pd.read_csv(DATA / "ubigeo_lookup.csv")
lu["dept_s"] = lu["region"].apply(strip)
lu["prov_s"] = lu["provincia"].apply(strip)
lu["dist_s"] = lu["distrito"].apply(strip)

gdf = gpd.read_file(DATA / "dist.shp")
gdf["dept_s"] = gdf["DEPARTAMEN"].apply(strip)
gdf["prov_s"] = gdf["PROVINCIA"].apply(strip)
gdf["dist_s"] = gdf["DISTRITO"].apply(strip)
gdf["shp_ubigeo"] = gdf["UBIGEO"].astype(str).str.zfill(6)

bridge = (
    lu.merge(
        gdf[["dept_s", "prov_s", "dist_s", "shp_ubigeo"]],
        on=["dept_s", "prov_s", "dist_s"],
        how="left",
    )[["id_ubigeo", "shp_ubigeo"]]
    .dropna(subset=["shp_ubigeo"])
)
print(f"  Bridged: {len(bridge):,} of {len(lu):,} lookup rows have a shapefile UBIGEO match")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — 2026 first-round results by district (keyed by shp_ubigeo)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading 2026 first-round results…")

meta = pd.read_parquet(
    DATA / "mesas_metadata.parquet",
    columns=["codigo_mesa", "id_ubigeo", "region", "provincia", "distrito",
             "codigo_estado_acta", "eleccion", "electores_habiles"],
)
pres_meta = meta[meta["eleccion"] == "presidencial"].copy()
pres_meta["id_ubigeo"] = pres_meta["id_ubigeo"].fillna(0).astype(int)

res = pd.read_parquet(
    DATA / "actas_resultados.parquet",
    columns=["codigo_mesa", "eleccion", "partido_nombre", "partido_codigo", "votos"],
)
pres_res = res[res["eleccion"] == "presidencial"].copy()

counted_set  = set(pres_meta[pres_meta["codigo_estado_acta"] == "C"]["codigo_mesa"])
meta_counted = pres_meta[pres_meta["codigo_estado_acta"] == "C"]

pres_counted = (
    pres_res[pres_res["codigo_mesa"].isin(counted_set)]
    .merge(pres_meta[["codigo_mesa", "id_ubigeo"]], on="codigo_mesa", how="left")
)

# Aggregate key parties by id_ubigeo
key_res = pres_counted[pres_counted["partido_nombre"].isin([KEIKO, SANCHEZ, NIETO, RLA])]
dist_votes = key_res.groupby(["id_ubigeo", "partido_nombre"])["votos"].sum().reset_index()

# Total valid votes per district (exclude blancos=80, nulos=81, impugnados=82)
valid = pres_counted[~pres_counted["partido_codigo"].isin(["80", "81", "82"])]
dist_total = valid.groupby("id_ubigeo")["votos"].sum().reset_index(name="total_valido")

dist_df = (
    dist_votes
    .pivot_table(index="id_ubigeo", columns="partido_nombre", values="votos", fill_value=0)
    .reset_index()
)
dist_df.columns.name = None
for p in [KEIKO, SANCHEZ, NIETO, RLA]:
    if p not in dist_df.columns:
        dist_df[p] = 0

# Mesa counts and electores
total_mesas   = pres_meta.groupby("id_ubigeo").size().reset_index(name="total_mesas")
counted_mesas = meta_counted.groupby("id_ubigeo").size().reset_index(name="counted_mesas")
electores     = pres_meta.groupby("id_ubigeo")["electores_habiles"].sum().reset_index(name="electores")

dist_df = (
    dist_df
    .merge(dist_total,    on="id_ubigeo", how="left")
    .merge(total_mesas,   on="id_ubigeo", how="left")
    .merge(counted_mesas, on="id_ubigeo", how="left")
    .merge(electores,     on="id_ubigeo", how="left")
)
dist_df["counted_mesas"] = dist_df["counted_mesas"].fillna(0).astype(int)

# Geo names for display
dist_df = dist_df.merge(
    lu[["id_ubigeo", "region", "provincia", "distrito"]], on="id_ubigeo", how="left"
)

# *** THE KEY STEP: translate id_ubigeo (ONPE) → shp_ubigeo (shapefile) ***
dist_df = dist_df.merge(bridge, on="id_ubigeo", how="left")
unmatched = dist_df["shp_ubigeo"].isna().sum()
print(f"  Districts without shapefile match: {unmatched} (will be excluded from map coloring)")

records = []
for _, r in dist_df.iterrows():
    u = str(r.get("shp_ubigeo") or "")
    if not u:
        continue
    records.append({
        "u":  u,                                           # shapefile UBIGEO — the map key
        "r":  str(r.get("region",   "") or ""),
        "p":  str(r.get("provincia","") or ""),
        "d":  str(r.get("distrito", "") or ""),
        "k":  round(float(r.get(KEIKO,   0) or 0)),       # Keiko first-round votes
        "s":  round(float(r.get(SANCHEZ, 0) or 0)),       # Sánchez first-round votes
        "n":  round(float(r.get(NIETO,   0) or 0)),       # Nieto first-round votes
        "rl": round(float(r.get(RLA,     0) or 0)),       # RLA first-round votes
        "tv": round(float(r.get("total_valido", 0) or 0)),# Total valid first-round votes
        "tm": int(r.get("total_mesas",   0) or 0),        # Total mesas in district
        "cm": int(r.get("counted_mesas", 0) or 0),        # Counted mesas (≈ all, from R1)
        "el": round(float(r.get("electores",   0) or 0)), # Electores hábiles
    })

out1 = DATA / "r1_districts.json"
out1.write_text(json.dumps(records, ensure_ascii=False, separators=(",", ":")))
total_k = sum(r["k"] for r in records)
total_s = sum(r["s"] for r in records)
total_v = sum(r["tv"] for r in records)
print(f"✅ r1_districts.json  — {len(records):,} districts, {out1.stat().st_size//1024} KB")
print(f"   Keiko:   {total_k:,}  ({total_k/total_v*100:.1f}% of valid)")
print(f"   Sánchez: {total_s:,}  ({total_s/total_v*100:.1f}% of valid)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — 2021 runoff results by district (keyed by shp_ubigeo)
#
# The 2021 CSV's UBIGEO column is already in the INEI system (same as dist.shp),
# so we can use it directly as shp_ubigeo — no extra bridging needed.
#
# VOTOS_P1 = Castillo (Peru Libre)   ← analog of Sánchez in this runoff
# VOTOS_P2 = Keiko (Fuerza Popular)  ← same candidate
# ─────────────────────────────────────────────────────────────────────────────
print("Loading 2021 runoff data…")

df21 = pd.read_csv(
    DATA / "Peruvian_Presidential_Election_Second_Round.csv",
    encoding="latin1", sep=";", index_col=False,
)
for col in ["VOTOS_P1", "VOTOS_P2", "VOTOS_VB", "VOTOS_VN"]:
    df21[col] = pd.to_numeric(df21[col], errors="coerce").fillna(0)

df21 = df21[df21["TIPO_ELECCION"] == "PRESIDENCIAL"].copy()
df21["shp_ubigeo"] = df21["UBIGEO"].astype(str).str.zfill(6)

dist21 = (
    df21.groupby("shp_ubigeo")
    .agg(castillo=("VOTOS_P1", "sum"), keiko21=("VOTOS_P2", "sum"))
    .reset_index()
)
dist21["tv21"] = dist21["castillo"] + dist21["keiko21"]
dist21 = dist21[dist21["tv21"] > 0]

records21 = [
    {
        "u":   str(r["shp_ubigeo"]),         # shapefile UBIGEO — same map key
        "c21": round(float(r["castillo"])),   # Castillo 2021 (analog of Sánchez)
        "k21": round(float(r["keiko21"])),    # Keiko 2021
        "tv21": round(float(r["tv21"])),
    }
    for _, r in dist21.iterrows()
]

out2 = DATA / "r2021_districts.json"
out2.write_text(json.dumps(records21, ensure_ascii=False, separators=(",", ":")))
total_c21 = sum(r["c21"] for r in records21)
total_k21 = sum(r["k21"] for r in records21)
print(f"✅ r2021_districts.json — {len(records21):,} districts, {out2.stat().st_size//1024} KB")
print(f"   Castillo: {total_c21:,}  ({total_c21/(total_c21+total_k21)*100:.1f}%)")
print(f"   Keiko 21: {total_k21:,}  ({total_k21/(total_c21+total_k21)*100:.1f}%)")

print("\nDone. Now run:  python generate_map_svg.py")
