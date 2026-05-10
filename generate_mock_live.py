"""
generate_mock_live.py
=====================
Generates data/live.json with simulated election results.

Usage:
  python3 generate_mock_live.py --pct 35              # tight race, 35% reported
  python3 generate_mock_live.py --keiko --pct 35      # Keiko winning ~52%
  python3 generate_mock_live.py --sanchez --pct 35    # Sanchez winning ~52%
  python3 generate_mock_live.py --random --pct 10     # pure random 50/50, no pattern
  python3 generate_mock_live.py --pct 0               # reset to empty
"""

import argparse, json, datetime
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")

parser = argparse.ArgumentParser()
parser.add_argument("--pct",     type=float, default=35.0)
parser.add_argument("--seed",    type=int,   default=42)
parser.add_argument("--keiko",   action="store_true")
parser.add_argument("--sanchez", action="store_true")
parser.add_argument("--random",  action="store_true", help="Pure random 50/50, no geographic pattern")
args = parser.parse_args()

if args.pct == 0:
    live = {"meta": {"total_mesas": 92766, "counted_mesas": 0, "pct_reported": 0.0,
                     "k_votes": 0, "s_votes": 0, "blancos": 0, "nulos": 0,
                     "valid_votes": 0, "timestamp": None, "live": False}, "districts": []}
    (DATA / "live.json").write_text(json.dumps(live, indent=2))
    print("✅ Reset to empty live.json")
    exit()

np.random.seed(args.seed)
PCT = args.pct / 100.0

if args.random:
    scenario = "Pure random 50/50"
elif args.keiko:
    scenario = "Keiko winning (~52%)"
elif args.sanchez:
    scenario = "Sánchez winning (~52%)"
else:
    scenario = "Tight race (~50/50)"

print(f"Generating mock live.json: {args.pct:.0f}% reported — {scenario}")

# ── Load R1 data ───────────────────────────────────────────────────────────────
r1 = json.loads((DATA / "r1_districts.json").read_text())
r1_map = {d["u"]: d for d in r1 if d.get("u") and d["u"] != "nan"}

# ── Pure random mode ───────────────────────────────────────────────────────────
if args.random:
    # Pick exactly PCT * 92766 mesas completely at random
    # No geographic bias, no pattern — just random tables
    all_mesas = []
    for u, d in r1_map.items():
        for _ in range(d["tm"]):
            all_mesas.append(u)

    n_total = len(all_mesas)
    n_cnt   = round(n_total * PCT)
    chosen  = np.random.choice(n_total, n_cnt, replace=False)
    chosen_set = set(chosen)

    # For each chosen mesa: assign votes randomly around 50/50
    # Each table has ~130 valid votes; Keiko gets random(40-60%)
    dist_votes = {}
    for idx in chosen_set:
        u = all_mesas[idx]
        d = r1_map[u]
        avg_vv = max(d["tv"] / max(d["tm"], 1), 50)
        vv = round(np.random.normal(avg_vv * 0.88, avg_vv * 0.05))
        vv = max(vv, 10)
        k_sh = np.random.uniform(0.38, 0.62)  # uniform between 38-62%
        k = round(vv * k_sh)
        s = vv - k
        bl = round(np.random.uniform(0.03, 0.08) * vv)
        nu = round(np.random.uniform(0.02, 0.06) * vv)
        if u not in dist_votes:
            dist_votes[u] = {"k": 0, "s": 0, "bl": 0, "nu": 0, "cm": 0, "tm": d["tm"]}
        dist_votes[u]["k"]  += k
        dist_votes[u]["s"]  += s
        dist_votes[u]["bl"] += bl
        dist_votes[u]["nu"] += nu
        dist_votes[u]["cm"] += 1

    districts_out = [{"u": u, **v} for u, v in dist_votes.items()]
    total_k  = sum(v["k"]  for v in dist_votes.values())
    total_s  = sum(v["s"]  for v in dist_votes.values())
    total_bl = sum(v["bl"] for v in dist_votes.values())
    total_nu = sum(v["nu"] for v in dist_votes.values())
    total_cm = sum(v["cm"] for v in dist_votes.values())
    valid    = total_k + total_s

else:
    # ── Geographic simulation ──────────────────────────────────────────────────
    if args.keiko:
        BASE_K = {'fp':0.92,'rla':0.72,'nieto':0.55,'others':0.42,'jpp':0.08}
    elif args.sanchez:
        BASE_K = {'fp':0.85,'rla':0.58,'nieto':0.42,'others':0.30,'jpp':0.05}
    else:
        BASE_K = {'fp':0.88,'rla':0.65,'nieto':0.48,'others':0.36,'jpp':0.07}

    DEPT_BIAS = {
        "15":1.5,"07":1.4,"04":1.2,"20":1.2,"21":1.1,"14":1.1,
        "06":0.8,"16":0.7,"10":0.7,"08":0.75,"05":0.8,
    }

    districts_out = []
    total_k = total_s = total_bl = total_nu = 0
    total_cm = 0

    for u, d in r1_map.items():
        tm = d["tm"]
        if tm == 0: continue
        dept = u[:2]
        bias = DEPT_BIAS.get(dept, 1.0)
        p = np.clip(PCT * bias + np.random.normal(0, 0.10), 0, 1)
        cm = min(tm, max(0, round(tm * p)))
        if cm == 0: continue
        tv = d["tv"]
        if tv == 0: continue
        fp_sh  = d["k"] / tv; jpp_sh = d["s"] / tv
        rla_sh = d["rl"] / tv; nieto_sh = d["n"] / tv
        other_sh = max(0, 1 - fp_sh - jpp_sh - rla_sh - nieto_sh)
        k_sh = np.clip(
            fp_sh*BASE_K['fp'] + rla_sh*BASE_K['rla'] +
            nieto_sh*BASE_K['nieto'] + other_sh*BASE_K['others'] +
            jpp_sh*BASE_K['jpp'] + np.random.normal(0, 0.04), 0.05, 0.95)
        avg_vv = tv / tm
        vv = round(avg_vv * cm * np.clip(np.random.normal(0.88, 0.04), 0.75, 1.05))
        blank_rate = np.clip(np.random.normal(0.08, 0.02), 0.04, 0.15)
        validos = round(vv * (1 - blank_rate))
        bl = round(vv * blank_rate * 0.55); nu = round(vv * blank_rate * 0.45)
        k = round(validos * k_sh); s = validos - k
        total_k += k; total_s += s; total_bl += bl; total_nu += nu; total_cm += cm
        districts_out.append({"u":u,"k":k,"s":s,"bl":bl,"nu":nu,"cm":cm,"tm":tm})

    valid = total_k + total_s

pct_rep = total_cm / 92766 * 100
k_pct   = total_k / valid * 100 if valid else 0
print(f"  Mesas: {total_cm:,} / 92,766 ({pct_rep:.1f}%)")
print(f"  Keiko:   {total_k:,} ({k_pct:.2f}%)")
print(f"  Sánchez: {total_s:,} ({100-k_pct:.2f}%)")
print(f"  Margen:  {total_k-total_s:+,} ({(total_k-total_s)/max(valid,1)*100:+.2f}%)")

live = {
    "meta": {
        "total_mesas":   92766,
        "counted_mesas": total_cm,
        "pct_reported":  round(pct_rep, 2),
        "k_votes":       total_k,
        "s_votes":       total_s,
        "blancos":       total_bl,
        "nulos":         total_nu,
        "valid_votes":   valid,
        "timestamp":     datetime.datetime.now().strftime("%d/%m/%Y %H:%M") + " (SIMULADO)",
        "live":          False,
    },
    "districts": districts_out,
}

(DATA / "live.json").write_text(json.dumps(live, ensure_ascii=False, separators=(",",":")))
print(f"✅  Saved data/live.json")
