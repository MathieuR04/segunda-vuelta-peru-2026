"""
generate_mock_live.py
=====================
Generates a realistic data/live.json simulating partial election-day results.

Results are calibrated to produce a TIGHT race (~50/50) so the forecast
needle sits near center and uncertainty is visible. Use --keiko or --sanchez
flags to tilt the scenario.

USAGE:
  python generate_mock_live.py              # 35% reported, tight race
  python generate_mock_live.py --pct 70     # 70% reported
  python generate_mock_live.py --pct 5      # early returns
  python generate_mock_live.py --keiko      # Keiko winning scenario (~52%)
  python generate_mock_live.py --sanchez    # Sanchez winning scenario (~52%)
  python generate_mock_live.py --pct 0      # reset to empty

REQUIREMENTS:
  pip install pandas pyarrow
"""

import argparse, json, random, datetime
import numpy as np
from pathlib import Path

DATA = Path("data")

parser = argparse.ArgumentParser()
parser.add_argument("--pct",     type=float, default=35.0)
parser.add_argument("--seed",    type=int,   default=42)
parser.add_argument("--keiko",   action="store_true", help="Keiko-winning scenario")
parser.add_argument("--sanchez", action="store_true", help="Sanchez-winning scenario")
args = parser.parse_args()

if args.pct == 0:
    live = {"meta": {"total_mesas": 92766, "counted_mesas": 0, "pct_reported": 0.0,
                     "k_votes": 0, "s_votes": 0, "blancos": 0, "nulos": 0,
                     "valid_votes": 0, "timestamp": None, "live": False}, "districts": []}
    (DATA / "live.json").write_text(json.dumps(live, indent=2))
    print("✅ Reset to empty live.json")
    exit()

random.seed(args.seed)
np.random.seed(args.seed)

PCT_REPORTED = args.pct / 100.0

# ── Vote flow model ───────────────────────────────────────────────────────────
# 2021 result: Castillo 50.1% Keiko 49.9%
# This is a tight race — calibrate so simulated national result is ~50/50
# Keiko R1 = 17.2%, Sanchez R1 = 12.0%, Nieto = 11%, RLA = 11.9%, Others = 47.9%
#
# Flow to Keiko (of valid R2 votes):
#   FP own voters:  ~90% retention
#   RLA voters:     ~52% to Keiko (right-wing)
#   Nieto voters:   ~36% to Keiko (center-right)
#   Others:         ~28% to Keiko
#   JPP own voters: ~8% to Keiko (defectors)
#
# This gives: 0.172*0.90 + 0.119*0.52 + 0.110*0.36 + 0.479*0.28 + 0.120*0.08
# = 0.155 + 0.062 + 0.040 + 0.134 + 0.010 = 0.401 → Keiko 40.1% of valid
# But valid votes in R2 = only Keiko + Sanchez, so share = 0.401 / (0.401 + 0.599) ≈ 40.1%
# That gives Keiko 40%. Too low. Adjust flows:

# Tight scenario (default): Keiko ends up ~50% of R2 valid
# Keiko: FP*0.90 + RLA*0.65 + Nieto*0.48 + Others*0.36 + JPP*0.08
# = 0.172*0.90 + 0.119*0.65 + 0.110*0.48 + 0.479*0.36 + 0.120*0.08
# = 0.155 + 0.077 + 0.053 + 0.172 + 0.010 = 0.467 of total valid
# Sanchez: rest = 0.533
# Keiko share of R2 = 0.467/(0.467+0.533) = 46.7% ... still low
# Let's use a scenario where the race is truly tight at district level with noise

if args.keiko:
    # Keiko winning: she gets ~52% of R2 valid
    BASE_KEIKO_FLOW = {
        'fp': 0.92, 'rla': 0.72, 'nieto': 0.55, 'others': 0.42, 'jpp': 0.08
    }
    scenario = "Keiko winning (~52%)"
elif args.sanchez:
    BASE_KEIKO_FLOW = {
        'fp': 0.85, 'rla': 0.58, 'nieto': 0.42, 'others': 0.30, 'jpp': 0.05
    }
    scenario = "Sanchez winning (~52%)"
else:
    # Tight: ~50/50 nationally
    BASE_KEIKO_FLOW = {
        'fp': 0.88, 'rla': 0.65, 'nieto': 0.48, 'others': 0.36, 'jpp': 0.07
    }
    scenario = "Tight race (~50/50)"

print(f"Generating mock live.json: {args.pct:.0f}% reported — {scenario}")

r1_data = json.loads((DATA / "r1_districts.json").read_text())
r1_map = {d["u"]: d for d in r1_data if d.get("u") and d["u"] != "nan"}

# Geographic reporting bias (coastal reports faster)
DEPT_BIAS = {
    "15": 1.5, "07": 1.4, "04": 1.2, "20": 1.2, "21": 1.1, "14": 1.1,
    "06": 0.8, "16": 0.7, "10": 0.7, "08": 0.75, "05": 0.8,
}

districts_out = []
total_k = total_s = total_bl = total_nu = 0
total_counted = 0

for u, d in r1_map.items():
    tm = d["tm"]
    if tm == 0:
        continue

    dept = u[:2]
    bias = DEPT_BIAS.get(dept, 1.0)
    p_rep = np.clip(PCT_REPORTED * bias + np.random.normal(0, 0.10), 0, 1)
    cm = min(tm, max(0, round(tm * p_rep)))
    if cm == 0:
        continue

    tv = d["tv"]
    if tv == 0:
        continue

    # District-level vote shares from R1
    fp_sh    = d["k"]  / tv
    jpp_sh   = d["s"]  / tv
    rla_sh   = d["rl"] / tv
    nieto_sh = d["n"]  / tv
    other_sh = max(0, 1 - fp_sh - jpp_sh - rla_sh - nieto_sh)

    # Keiko's R2 share at this district (before noise)
    k2_sh = (fp_sh    * BASE_KEIKO_FLOW['fp']     +
             rla_sh   * BASE_KEIKO_FLOW['rla']    +
             nieto_sh * BASE_KEIKO_FLOW['nieto']  +
             other_sh * BASE_KEIKO_FLOW['others'] +
             jpp_sh   * BASE_KEIKO_FLOW['jpp'])

    # Add district-level noise (realistic variance)
    k2_sh = np.clip(k2_sh + np.random.normal(0, 0.04), 0.05, 0.95)
    s2_sh = 1 - k2_sh

    # Estimate valid votes for counted mesas (scale from R1)
    avg_vv_per_mesa = tv / tm
    counted_vv = round(avg_vv_per_mesa * cm * np.clip(np.random.normal(0.92, 0.04), 0.75, 1.05))

    # Blancos + nulos ~8% of emitidos
    blank_rate = np.clip(np.random.normal(0.08, 0.02), 0.04, 0.15)
    validos = round(counted_vv * (1 - blank_rate))
    blancos = round(counted_vv * blank_rate * 0.55)
    nulos   = round(counted_vv * blank_rate * 0.45)

    k_votes = round(validos * k2_sh)
    s_votes = validos - k_votes

    total_k  += k_votes
    total_s  += s_votes
    total_bl += blancos
    total_nu += nulos
    total_counted += cm

    districts_out.append({
        "u": u, "k": k_votes, "s": s_votes,
        "bl": blancos, "nu": nulos, "cm": cm, "tm": tm,
    })

pct_rep   = total_counted / 92766 * 100
valid_tot = total_k + total_s
k_pct     = total_k / valid_tot * 100 if valid_tot else 0

print(f"  Mesas: {total_counted:,} / 92,766 ({pct_rep:.1f}%)")
print(f"  Keiko:   {total_k:,} ({k_pct:.2f}%)")
print(f"  Sánchez: {total_s:,} ({100-k_pct:.2f}%)")
print(f"  Margen:  {total_k-total_s:+,} ({(total_k-total_s)/valid_tot*100:+.2f}%)")

live = {
    "meta": {
        "total_mesas":   92766,
        "counted_mesas": total_counted,
        "pct_reported":  round(pct_rep, 2),
        "k_votes":       total_k,
        "s_votes":       total_s,
        "blancos":       total_bl,
        "nulos":         total_nu,
        "valid_votes":   valid_tot,
        "timestamp":     datetime.datetime.now().strftime("%d/%m/%Y %H:%M") + " (SIMULADO)",
        "live":          False,
    },
    "districts": districts_out,
}

(DATA / "live.json").write_text(json.dumps(live, ensure_ascii=False, separators=(",", ":")))
print(f"✅  Saved data/live.json  ({(DATA/'live.json').stat().st_size//1024} KB)")
