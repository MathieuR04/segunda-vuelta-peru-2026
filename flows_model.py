"""
flows_model.py — Segunda Vuelta Perú 2026
==========================================
Pure table-level OLS:
  keiko_r2(t) = β₀ + β_fp*fp_sh(t) + β_jpp*jpp_sh(t) + β_rla*rla_sh(t) + ...
  sanchez_r2(t) = same
  etc.

Each observation is one polling table. R1 party shares are regressors.
R2 outcomes come from live.json district aggregates (assigned to each table).
Bootstrap 200x for CIs.
"""

import json, warnings, datetime
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA        = Path('data')
N_BOOTSTRAP = 200

PARTIES = {
    '00000008': 'fp',
    '00000010': 'jpp',
    '00000035': 'rla',
    '00000016': 'nieto',
    '00000014': 'obras',
    '00000023': 'ppt',
    '00000002': 'an',
}
PARTY_NAMES  = {'fp':'Fuerza Popular (Keiko Fujimori)','jpp':'Juntos por el Perú (Roberto Sánchez)',
                'rla':'Renovación Popular (RLA)','nieto':'Partido del Buen Gobierno (Nieto)',
                'obras':'Partido Cívico Obras','ppt':'País para Todos','an':'Ahora Nación'}
PARTY_COLORS = {'fp':'#f57b2d','jpp':'#48cb50','rla':'#1e3a8a',
                'nieto':'#d97706','obras':'#7c3aed','ppt':'#0891b2','an':'#be185d'}
SOURCE_PARTIES = list(PARTIES.values())
DEST_LABELS    = ['keiko','sanchez','blanco','nulo','no_vota']
DEST_NAMES     = ['Keiko','Sánchez','Blanco','Nulo','No vota']
DEST_COLORS    = ['#f57b2d','#48cb50','#b0a898','#888888','#cccccc']

print("=" * 60)
print("VOTE FLOWS MODEL — Segunda Vuelta Perú 2026")
print("=" * 60)

# ── Load metadata + R1 ───────────────────────────────────────────────────────
print("\n[1/4] Loading R1 table-level data...")
meta = pd.read_parquet(DATA / 'mesas_metadata.parquet',
    columns=['codigo_mesa','id_ubigeo','eleccion','electores_habiles'])
meta = meta[meta['eleccion'] == 'presidencial'].copy()
meta['id_ubigeo']  = meta['id_ubigeo'].fillna(0).astype(int)
meta['dist']       = meta['id_ubigeo'].astype(int)
meta['electores']  = meta['electores_habiles'].fillna(295).clip(lower=10)

res = pd.read_parquet(DATA / 'actas_resultados.parquet',
    columns=['codigo_mesa','eleccion','partido_codigo','votos'])
res = res[res['eleccion'] == 'presidencial'].copy()
res_k = res[res['partido_codigo'].isin(PARTIES)].copy()
res_k['feat'] = res_k['partido_codigo'].map(PARTIES)
r1w = (res_k.pivot_table(index='codigo_mesa', columns='feat',
    values='votos', fill_value=0).reset_index())
r1w.columns.name = None
for p in SOURCE_PARTIES:
    if p not in r1w.columns: r1w[p] = 0

tv_r1 = (res[~res['partido_codigo'].isin(['80','81','82'])]
    .groupby('codigo_mesa')['votos'].sum().reset_index(name='tv_r1'))
# Also get blancos and nulos from R1 (control regressors)
bl_r1 = (res[res['partido_codigo']=='80']
    .groupby('codigo_mesa')['votos'].sum().reset_index(name='bl_r1'))
nu_r1 = (res[res['partido_codigo']=='81']
    .groupby('codigo_mesa')['votos'].sum().reset_index(name='nu_r1'))
r1w = r1w.merge(tv_r1, on='codigo_mesa', how='left')
r1w = r1w.merge(bl_r1, on='codigo_mesa', how='left')
r1w = r1w.merge(nu_r1, on='codigo_mesa', how='left')
r1w['tv_r1']  = r1w['tv_r1'].fillna(0)
r1w['bl_r1']  = r1w['bl_r1'].fillna(0)
r1w['nu_r1']  = r1w['nu_r1'].fillna(0)

df = meta.merge(r1w, on='codigo_mesa', how='left')
for p in SOURCE_PARTIES:
    if p not in df.columns: df[p] = 0
    df[p] = df[p].fillna(0)
df['tv_r1']  = df['tv_r1'].fillna(0)
df['bl_r1']  = df['bl_r1'].fillna(0)
df['nu_r1']  = df['nu_r1'].fillna(0)
df = df[df['tv_r1'] > 0].copy()

# Total emitidos R1 (valid + blancos + nulos, excl impugnados)
df['ta_r1'] = df['tv_r1'] + df['bl_r1'] + df['nu_r1']

# R1 party shares of valid votes (the 7 main parties — for display)
SOURCES = [f'{p}_sh' for p in SOURCE_PARTIES]
for p in SOURCE_PARTIES:
    df[f'{p}_sh'] = df[p] / df['tv_r1'].clip(lower=1)

# Additional control regressors: R1 blancos, nulos, abstention, other minor parties
# "other" = valid votes not accounted for by the 7 main parties
df['other_r1'] = np.clip(df['tv_r1'] - sum(df[p] for p in SOURCE_PARTIES), 0, None)
df['bl_r1_sh']    = df['bl_r1']    / df['ta_r1'].clip(lower=1)
df['nu_r1_sh']    = df['nu_r1']    / df['ta_r1'].clip(lower=1)
df['other_r1_sh'] = df['other_r1'] / df['tv_r1'].clip(lower=1)
df['abs_r1_sh']   = np.clip(1 - df['ta_r1'] / df['electores'].clip(lower=1), 0, 1)

# All regressors: 7 main + 4 controls (blancos, nulos, other, abstention)
SOURCES_ALL = SOURCES + ['bl_r1_sh', 'nu_r1_sh', 'other_r1_sh', 'abs_r1_sh']

print(f"  Total tables with R1 data: {len(df):,}")

# ── Assign R2 outcomes to each table ─────────────────────────────────────────
print("[2/4] Assigning R2 outcomes...")

live_raw_path = DATA / 'live_raw.parquet'
if live_raw_path.exists():
    # Table-level R2 — ideal
    lr = pd.read_parquet(live_raw_path)
    lr = lr[lr['estado'] == 'C'].copy()
    for c in ['k_votes','s_votes','blancos','nulos']:
        lr[c] = pd.to_numeric(lr[c], errors='coerce').fillna(0)
    lr['ta_r2'] = lr['k_votes'] + lr['s_votes'] + lr['blancos'] + lr['nulos']
    df = df.merge(lr[['codigo_mesa','k_votes','s_votes','blancos','nulos','ta_r2']],
                  on='codigo_mesa', how='inner')
    print(f"  Table-level R2: {len(df):,} counted tables")
else:
    # District-aggregate fallback: assign district R2 shares to each table
    # R2 outcomes are at table level via electores denominator
    import unicodedata, geopandas as gpd
    def strip(s):
        if not isinstance(s, str): return ""
        return ''.join(c for c in unicodedata.normalize("NFD", s)
                       if unicodedata.category(c) != 'Mn').upper().strip()
    live = json.loads((DATA / 'live.json').read_text())
    lByU = {d['u']: d for d in live.get('districts', [])}
    lu = pd.read_csv(DATA / 'ubigeo_lookup.csv')
    lu['r_s'] = lu['region'].apply(strip); lu['p_s'] = lu['provincia'].apply(strip); lu['d_s'] = lu['distrito'].apply(strip)
    gdf = gpd.read_file(DATA / 'dist.shp')
    gdf['r_s'] = gdf['DEPARTAMEN'].apply(strip); gdf['p_s'] = gdf['PROVINCIA'].apply(strip); gdf['d_s'] = gdf['DISTRITO'].apply(strip)
    gdf['shp_ubigeo'] = gdf['UBIGEO'].astype(str).str.zfill(6)
    bridge = lu.merge(gdf[['r_s','p_s','d_s','shp_ubigeo']], on=['r_s','p_s','d_s'], how='left')[['id_ubigeo','shp_ubigeo']].dropna()
    id2shp = bridge.set_index('id_ubigeo')['shp_ubigeo'].to_dict()
    df['shp_ubigeo'] = df['id_ubigeo'].map(id2shp)

    n_cnt = live.get('meta', {}).get('counted_mesas', 0)
    df['cm_frac'] = df['shp_ubigeo'].map(
        lambda u: lByU[str(u)]['cm'] / max(lByU[str(u)]['tm'], 1)
                  if pd.notna(u) and str(u) in lByU else 0.0)
    df_s = df.sort_values('cm_frac', ascending=False).reset_index(drop=True)
    df_s['is_cnt'] = df_s.index < n_cnt
    df = df_s[df_s['is_cnt']].copy()

    # R2 votes: scale district totals proportionally to table's R1 size
    def get_d(u): return lByU.get(str(u)) if pd.notna(u) else None
    df['_d'] = df['shp_ubigeo'].map(get_d)
    df['_tm'] = df['_d'].map(lambda d: max(d['tm'], 1) if d else 1)
    for col, key in [('k_votes','k'),('s_votes','s'),('blancos','bl'),('nulos','nu')]:
        df[col] = df['_d'].map(lambda d: d[key] if d else 0) / df['_tm']
    df['ta_r2'] = df['k_votes'] + df['s_votes'] + df['blancos'] + df['nulos']
    df = df[df['ta_r2'] > 0].copy()
    print(f"  District fallback: {len(df):,} counted tables")

if len(df) < 100:
    print("  Too few tables.")
    (DATA / 'flows.json').write_text(json.dumps({'error': 'insufficient_data'}, indent=2))
    exit()

# ── R2 outcomes as shares of electores ───────────────────────────────────────
df['el'] = df['electores'].clip(lower=50)
df['to_keiko']   = (df['k_votes']  / df['el']).clip(0, 1)
df['to_sanchez'] = (df['s_votes']  / df['el']).clip(0, 1)
df['to_blanco']  = (df['blancos']  / df['el']).clip(0, 1)
df['to_nulo']    = (df['nulos']    / df['el']).clip(0, 1)
df['to_no_vota'] = np.clip(1 - df['ta_r2'] / df['el'], 0, 1)

DESTS = [f'to_{d}' for d in DEST_LABELS]
df = df.dropna(subset=SOURCES + DESTS)
df['w'] = np.sqrt(df['tv_r1'].clip(lower=1))   # weight by table size

print(f"  Tables for regression: {len(df):,}")
print(f"  Districts covered: {df['dist'].nunique():,}")

# ── Table-level OLS ───────────────────────────────────────────────────────────
print("[3/4] Fitting table-level OLS...")

X = np.column_stack([np.ones(len(df))] + [df[s].values for s in SOURCES_ALL])
w = df['w'].values

def wls(X, y, w):
    Xw = X * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return coef

main_coefs = {}
for dest in DESTS:
    y = df[dest].values
    coef = wls(X, y, w)
    main_coefs[dest] = coef
    y_pred = X @ coef
    r2 = float(1 - np.var(y - y_pred) / np.var(y)) if np.var(y) > 0 else 0
    print(f"  {dest}: R²={max(r2,0):.3f}")

# ── Bootstrap CIs ─────────────────────────────────────────────────────────────
print("[4/4] Bootstrap CIs...")
n = len(df)
boot = {dest: [] for dest in DESTS}
for b in range(N_BOOTSTRAP):
    idx = np.random.choice(n, n, replace=True)
    Xb = X[idx]; wb = w[idx]
    for dest in DESTS:
        yb = df[dest].values[idx]
        boot[dest].append(wls(Xb, yb, wb))
    if (b + 1) % 100 == 0:
        print(f"  Bootstrap: {b+1}/{N_BOOTSTRAP}")

# ── Extract flows via counterfactual prediction ───────────────────────────────
# For party s (among top 7): predict when that party has 100% of R1 valid votes,
# all other top-7 parties = 0, controls (blancos, nulos, other, abstention) = their means.
# Position of party s in X is i+1 (after intercept).
# Control columns are at positions len(SOURCE_PARTIES)+1 .. len(SOURCES_ALL).

n_main = len(SOURCE_PARTIES)
n_feats = X.shape[1]  # 1 + len(SOURCES_ALL)

# Mean values of control regressors (to hold fixed in counterfactual)
ctrl_means = {s: float(df[s].mean()) for s in ['bl_r1_sh','nu_r1_sh','other_r1_sh','abs_r1_sh']}
ctrl_idx = {s: n_main + 1 + j for j, s in enumerate(['bl_r1_sh','nu_r1_sh','other_r1_sh','abs_r1_sh'])}

# ── Extract flows via counterfactual prediction ───────────────────────────────
# For party s: what does the model predict when a table has 100% of votes for s?
# x_s = [1, 0, ..., 1, ..., 0] (intercept=1, party s = 1, others = 0)
# This gives interpretable flows: "if all R1 votes in a table were for party s,
# what fraction of electores would go to each R2 destination?"
# Then normalize the prediction vector to sum to 1 → probability distribution.

# Baseline: x_base = [1, 0, 0, ..., 0] (intercept only, all party shares = 0)
n_feats = X.shape[1]  # 1 intercept + n_parties
x_base = np.zeros(n_feats); x_base[0] = 1.0

flows_out = []
for i, party in enumerate(SOURCE_PARTIES):
    row = {'party': party, 'name': PARTY_NAMES[party], 'color': PARTY_COLORS[party]}

    # Counterfactual: party s = 1, all other top-7 = 0, controls at their means
    x_party = x_base.copy(); x_party[i + 1] = 1.0
    # Set controls to their mean values
    for s, idx_c in ctrl_idx.items():
        x_party[idx_c] = ctrl_means[s]

    pred = np.array([float(x_party @ main_coefs[dest]) for dest in DESTS])
    pred = np.clip(pred, 0, 1)
    total = pred.sum()
    flows = pred / total if total > 1e-10 else np.ones(len(DESTS)) / len(DESTS)

    for j, (dest, label) in enumerate(zip(DESTS, DEST_LABELS)):
        row[f'to_{label}'] = round(float(flows[j]), 4)
        boot_preds = [float(x_party @ boot[dest][b]) for b in range(N_BOOTSTRAP)]
        row[f'ci_{label}_lo'] = round(float(np.percentile(boot_preds, 5)), 4)
        row[f'ci_{label}_hi'] = round(float(np.percentile(boot_preds, 95)), 4)

    flows_out.append(row)
    print(f"  {party:8s}: K={row['to_keiko']:.2f} S={row['to_sanchez']:.2f} "
          f"B={row['to_blanco']:.2f} N={row['to_nulo']:.2f} A={row['to_no_vota']:.2f}")

(DATA / 'flows.json').write_text(json.dumps({
    "meta": {
        "timestamp":   datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "n_tables":    int(len(df)),
        "n_districts": int(df['dist'].nunique()),
        "method":      "table_level_OLS",
        "n_bootstrap": N_BOOTSTRAP,
    },
    "destinations": DEST_LABELS,
    "dest_names":   DEST_NAMES,
    "dest_colors":  DEST_COLORS,
    "parties":      flows_out,
}, ensure_ascii=False, indent=2))
print(f"\n✅  Saved: data/flows.json")
