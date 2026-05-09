"""
generate_mock_live.py
=====================
Generates a realistic data/live.json simulating partial election-day results.

Uses actual first-round data to simulate plausible second-round outcomes:
- Keiko gets ~her R1 share + some votes from other right-leaning parties
- Sanchez gets ~his R1 share + some votes from other left-leaning parties
- Realistic partial reporting (configurable %)

Run from project root:
  python generate_mock_live.py             # default: 35% reported
  python generate_mock_live.py --pct 70   # 70% reported
  python generate_mock_live.py --pct 5    # early returns only

REQUIREMENTS:
  pip install pandas pyarrow
"""

import argparse, json, random, datetime
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")

parser = argparse.ArgumentParser()
parser.add_argument("--pct", type=float, default=35.0, help="% of mesas reported (0-100)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

PCT_REPORTED = args.pct / 100.0
print(f"Generating mock live.json with {args.pct:.0f}% mesas reported...")

# ── Load first-round data ─────────────────────────────────────────────────────
r1 = json.loads((DATA / "r1_districts.json").read_text())
r1_map = {d["u"]: d for d in r1 if d.get("u") and d["u"] != "nan"}

# National R1 totals
total_k_r1 = sum(d["k"] for d in r1)
total_s_r1 = sum(d["s"] for d in r1)
total_tv_r1 = sum(d["tv"] for d in r1)

print(f"  R1 — Keiko: {total_k_r1:,} ({total_k_r1/total_tv_r1*100:.1f}%)")
print(f"  R1 — Sánchez: {total_s_r1:,} ({total_s_r1/total_tv_r1*100:.1f}%)")

# ── Vote flow model ───────────────────────────────────────────────────────────
# Runoff = Keiko vs. Sánchez only. Votes from other R1 parties redistribute.
# This is a plausible scenario: Keiko gains from Nieto/Obras/some others
# Sanchez gains from Ahora Nacion/Perú Libre voters

# At district level, simulate R2 results:
# keiko_r2_share = f(keiko_r1_share, nieto_r1_share, rla_r1_share, ...)
# Use a simple linear model:
#   keiko_r2 = keiko_r1 * 0.95      (some defectors)
#            + nieto_r1 * 0.38      (moderate right)
#            + rla_r1   * 0.55      (right-wing)
#            + others   * 0.28      (spread)
#   sanchez_r2 = rest
# Then add district-level noise

def simulate_district_r2(d):
    tv = d["tv"]
    if tv == 0:
        return 0, 0, 0, 0
    k_r1_sh  = d["k"]  / tv
    s_r1_sh  = d["s"]  / tv
    n_r1_sh  = d["n"]  / tv   # Nieto
    rl_r1_sh = d["rl"] / tv   # RLA
    other_sh = 1 - k_r1_sh - s_r1_sh - n_r1_sh - rl_r1_sh

    # Keiko R2 share of valid votes
    keiko_r2_sh = (
        k_r1_sh  * 0.93 +
        n_r1_sh  * 0.38 +
        rl_r1_sh * 0.55 +
        other_sh * 0.27
    )
    # Add district-level noise
    keiko_r2_sh = np.clip(keiko_r2_sh + np.random.normal(0, 0.03), 0.05, 0.95)
    sanchez_r2_sh = 1 - keiko_r2_sh

    # Estimate turnout slightly higher than R1 (runoff typically lower, but let's use similar)
    electores = d["el"]
    turnout = np.clip(np.random.normal(0.75, 0.05), 0.55, 0.90) if electores > 0 else 0.75
    total_emitidos = round(electores * turnout) if electores > 0 else tv
    
    # Blancos + nulos ~= 8% of emitidos in runoff
    blank_nulo_rate = np.clip(np.random.normal(0.08, 0.02), 0.03, 0.18)
    validos = round(total_emitidos * (1 - blank_nulo_rate))
    blancos = round(total_emitidos * blank_nulo_rate * 0.55)
    nulos   = round(total_emitidos * blank_nulo_rate * 0.45)

    k_votes = round(validos * keiko_r2_sh)
    s_votes = validos - k_votes

    return k_votes, s_votes, blancos, nulos

# ── Decide which mesas are "reported" ────────────────────────────────────────
# Simulate geographic bias: coastal + Lima report faster
# Use R1 dept code to weight reporting probability

DEPT_REPORT_BIAS = {
    # Coastal/Lima departments report faster
    "15": 1.6,  # Lima
    "07": 1.4,  # Callao
    "04": 1.3,  # Arequipa
    "14": 1.3,  # San Martin (surprise - often early)
    "06": 1.2,  # Cajamarca — actually slow but simulate
    "20": 1.3,  # Tacna
    "21": 1.1,  # Tumbes
    "17": 1.1,  # Puno
    # Highland/jungle slower
    "16": 0.7,  # Pasco
    "10": 0.7,  # Huánuco
    "08": 0.8,  # Huancavelica
    "05": 0.8,  # Ayacucho
}

districts_out = []
total_k = total_s = total_bl = total_nu = total_vv = 0
total_counted = 0

for u, d in r1_map.items():
    dept_code = u[:2]
    bias = DEPT_REPORT_BIAS.get(dept_code, 1.0)
    
    tm = d["tm"]
    if tm == 0:
        continue

    # How many mesas in this district are reported?
    p_report = np.clip(PCT_REPORTED * bias + np.random.normal(0, 0.08), 0, 1)
    cm = round(tm * p_report)
    cm = max(0, min(tm, cm))

    if cm == 0:
        continue

    # Simulate R2 result for this district (based on counted fraction)
    k2, s2, bl2, nu2 = simulate_district_r2(d)
    
    # Scale to counted mesas
    scale = cm / tm
    k_c  = round(k2  * scale)
    s_c  = round(s2  * scale)
    bl_c = round(bl2 * scale)
    nu_c = round(nu2 * scale)

    total_k  += k_c
    total_s  += s_c
    total_bl += bl_c
    total_nu += nu_c
    total_vv += k_c + s_c
    total_counted += cm

    districts_out.append({
        "u":  u,
        "k":  k_c,
        "s":  s_c,
        "bl": bl_c,
        "nu": nu_c,
        "cm": cm,
        "tm": tm,
    })

pct_rep = total_counted / 92766 * 100
valid_total = total_k + total_s

print(f"\nResults:")
print(f"  Mesas counted: {total_counted:,} / 92,766 ({pct_rep:.1f}%)")
print(f"  Keiko:   {total_k:,} ({total_k/valid_total*100:.2f}%)")
print(f"  Sánchez: {total_s:,} ({total_s/valid_total*100:.2f}%)")
print(f"  Margen:  {total_k-total_s:+,} ({(total_k-total_s)/valid_total*100:+.2f}%)")

live = {
    "meta": {
        "total_mesas":   92766,
        "counted_mesas": total_counted,
        "pct_reported":  round(pct_rep, 2),
        "k_votes":       total_k,
        "s_votes":       total_s,
        "blancos":       total_bl,
        "nulos":         total_nu,
        "valid_votes":   valid_total,
        "timestamp":     datetime.datetime.now().strftime("%d/%m/%Y %H:%M") + " (SIMULADO)",
        "live":          False,
    },
    "districts": districts_out,
}

out = DATA / "live.json"
out.write_text(json.dumps(live, ensure_ascii=False, separators=(",", ":")))
print(f"\n✅  Saved {out}  ({out.stat().st_size // 1024} KB)")
print(f"\nOpen the site to see the simulation.")
print(f"To restore blank state: python generate_mock_live.py --pct 0")
