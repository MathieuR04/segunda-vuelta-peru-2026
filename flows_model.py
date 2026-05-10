"""
flows_model.py — Segunda Vuelta Perú 2026
==========================================
Five OLS regressions, one per R2 outcome:

  keiko_r2(t)   = β1·fp_r1(t) + β2·jpp_r1(t) + ... + βn·ausentes_r1(t) + ε
  sanchez_r2(t) = β1·fp_r1(t) + β2·jpp_r1(t) + ... + βn·ausentes_r1(t) + ε
  blanco_r2(t)  = ...
  nulo_r2(t)    = ...
  ausentes_r2(t)= ...

All variables in VOTE COUNTS (levels), not shares.
β_ij = estimated votes going from R1 party i to R2 destination j per vote.
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

# R1 regressors: all parties + blancos + nulos + ausentes
R1_PARTIES = {
    '00000008': 'fp',
    '00000010': 'jpp',
    '00000035': 'rla',
    '00000016': 'nieto',
    '00000014': 'obras',
    '00000023': 'ppt',
    '00000002': 'an',
    '00000001': 'app',   # Alianza para el Progreso
    '00000032': 'podemos',
    '00000033': 'primero',
    '80': 'blanco_r1',
    '81': 'nulo_r1',
}
# Top 7 to display in output
DISPLAY_PARTIES = ['fp','jpp','rla','nieto','obras','ppt','an']
PARTY_NAMES  = {
    'fp':'Fuerza Popular (Keiko Fujimori)', 'jpp':'Juntos por el Perú (Roberto Sánchez)',
    'rla':'Renovación Popular (RLA)',        'nieto':'Partido del Buen Gobierno (Nieto)',
    'obras':'Partido Cívico Obras',          'ppt':'País para Todos',
    'an':'Ahora Nación',
}
PARTY_COLORS = {
    'fp':'#f57b2d','jpp':'#48cb50','rla':'#1e3a8a',
    'nieto':'#d97706','obras':'#7c3aed','ppt':'#0891b2','an':'#be185d',
}

# R2 outcomes
DEST_LABELS = ['keiko','sanchez','blanco','nulo','ausentes']
DEST_NAMES  = ['Keiko','Sánchez','Blanco','Nulo','Ausentes']
DEST_COLORS = ['#f57b2d','#48cb50','#b0a898','#888888','#cccccc']

print("=" * 60)
print("VOTE FLOWS MODEL — Segunda Vuelta Perú 2026")
print("=" * 60)

# ── Load R1 vote counts per table ─────────────────────────────────────────────
print("\n[1/4] Loading R1 table-level vote counts...")
meta = pd.read_parquet(DATA / 'mesas_metadata.parquet',
    columns=['codigo_mesa','id_ubigeo','eleccion','electores_habiles'])
meta = meta[meta['eleccion'] == 'presidencial'].copy()
meta['id_ubigeo'] = meta['id_ubigeo'].fillna(0).astype(int)
meta['electores'] = meta['electores_habiles'].fillna(295).clip(lower=10)

res = pd.read_parquet(DATA / 'actas_resultados.parquet',
    columns=['codigo_mesa','eleccion','partido_codigo','votos'])
res = res[res['eleccion'] == 'presidencial'].copy()

# All named parties
res_k = res[res['partido_codigo'].isin(R1_PARTIES)].copy()
res_k['feat'] = res_k['partido_codigo'].map(R1_PARTIES)
r1w = res_k.pivot_table(index='codigo_mesa', columns='feat',
    values='votos', fill_value=0).reset_index()
r1w.columns.name = None
for p in R1_PARTIES.values():
    if p not in r1w.columns: r1w[p] = 0

# "otros" = all valid votes not in named parties
tv_r1 = (res[~res['partido_codigo'].isin(['80','81','82'])]
    .groupby('codigo_mesa')['votos'].sum().reset_index(name='tv_r1'))
r1w = r1w.merge(tv_r1, on='codigo_mesa', how='left')
r1w['tv_r1'] = r1w['tv_r1'].fillna(0)
named_valid = sum(r1w[p] for p in R1_PARTIES.values()
                  if p not in ['blanco_r1','nulo_r1'])
r1w['otros_r1'] = (r1w['tv_r1'] - named_valid).clip(lower=0)

# Ausentes R1 = electores - emitidos
ta_r1 = res.groupby('codigo_mesa')['votos'].sum().reset_index(name='ta_r1')
df = meta.merge(r1w, on='codigo_mesa', how='left')
df = df.merge(ta_r1, on='codigo_mesa', how='left')
df['ta_r1']      = df['ta_r1'].fillna(0)
df['ausentes_r1'] = (df['electores'] - df['ta_r1']).clip(lower=0)

ALL_R1 = DISPLAY_PARTIES + ['blanco_r1','nulo_r1','otros_r1','ausentes_r1']
for p in ALL_R1:
    if p not in df.columns: df[p] = 0
    df[p] = df[p].fillna(0)

df = df[df['tv_r1'] > 0].copy()
print(f"  Tables with R1 data: {len(df):,}")

# ── Load R2 outcomes ──────────────────────────────────────────────────────────
print("[2/4] Loading R2 outcomes...")
live_raw_path = DATA / 'live_raw.parquet'
if live_raw_path.exists():
    lr = pd.read_parquet(live_raw_path)
    lr = lr[lr['estado'] == 'C'].copy()
    for c in ['k_votes','s_votes','blancos','nulos']:
        lr[c] = pd.to_numeric(lr[c], errors='coerce').fillna(0)
    df = df.merge(lr[['codigo_mesa','k_votes','s_votes','blancos','nulos']],
                  on='codigo_mesa', how='inner')
    print(f"  Table-level R2: {len(df):,} tables")
else:
    import unicodedata, geopandas as gpd
    def strip(s):
        if not isinstance(s, str): return ""
        return ''.join(c for c in unicodedata.normalize("NFD", s)
                       if unicodedata.category(c) != 'Mn').upper().strip()
    live = json.loads((DATA / 'live.json').read_text())
    lByU = {d['u']: d for d in live.get('districts', [])}
    lu = pd.read_csv(DATA / 'ubigeo_lookup.csv')
    lu['r_s']=lu['region'].apply(strip); lu['p_s']=lu['provincia'].apply(strip); lu['d_s']=lu['distrito'].apply(strip)
    gdf = gpd.read_file(DATA / 'dist.shp')
    gdf['r_s']=gdf['DEPARTAMEN'].apply(strip); gdf['p_s']=gdf['PROVINCIA'].apply(strip); gdf['d_s']=gdf['DISTRITO'].apply(strip)
    gdf['shp_ubigeo'] = gdf['UBIGEO'].astype(str).str.zfill(6)
    bridge = lu.merge(gdf[['r_s','p_s','d_s','shp_ubigeo']], on=['r_s','p_s','d_s'], how='left')[['id_ubigeo','shp_ubigeo']].dropna()
    id2shp = bridge.set_index('id_ubigeo')['shp_ubigeo'].to_dict()
    df['shp_ubigeo'] = df['id_ubigeo'].map(id2shp)
    n_cnt = live.get('meta',{}).get('counted_mesas', 0)
    df['cm_frac'] = df['shp_ubigeo'].map(
        lambda u: lByU[str(u)]['cm']/max(lByU[str(u)]['tm'],1)
                  if pd.notna(u) and str(u) in lByU else 0.0)
    df_s = df.sort_values('cm_frac', ascending=False).reset_index(drop=True)
    df_s['is_cnt'] = df_s.index < n_cnt
    df = df_s[df_s['is_cnt']].copy()
    for col, key in [('k_votes','k'),('s_votes','s'),('blancos','bl'),('nulos','nu')]:
        df[col] = df['shp_ubigeo'].map(
            lambda u: lByU[str(u)][key]/max(lByU[str(u)]['tm'],1)
                      if pd.notna(u) and str(u) in lByU else 0)
    print(f"  District fallback: {len(df):,} tables")

if len(df) < 100:
    print("  Too few tables.")
    (DATA/'flows.json').write_text(json.dumps({'error':'insufficient_data'}, indent=2))
    exit()

# R2 ausentes = electores - emitidos_r2
df['emitidos_r2'] = df['k_votes'] + df['s_votes'] + df['blancos'] + df['nulos']
df['ausentes_r2'] = (df['electores'] - df['emitidos_r2']).clip(lower=0)
df = df[df['emitidos_r2'] > 0].copy()

print(f"  Tables for regression: {len(df):,}")

# ── OLS: 5 regressions, one per R2 outcome ────────────────────────────────────
print("[3/4] Fitting OLS (5 regressions × 1 per R2 outcome)...")

# X matrix: R1 vote counts for all parties (no intercept — votes must go somewhere)
X = df[ALL_R1].values  # shape (n_tables, n_r1_parties)

# R2 outcomes in vote counts
Y = {
    'keiko':   df['k_votes'].values,
    'sanchez': df['s_votes'].values,
    'blanco':  df['blancos'].values,
    'nulo':    df['nulos'].values,
    'ausentes':df['ausentes_r2'].values,
}

# Weight by electores (larger tables get more weight)
w = np.sqrt(df['electores'].clip(lower=1).values)

def wls(X, y, w):
    Xw = X * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return coef

main_coefs = {}
for dest in DEST_LABELS:
    coef = wls(X, Y[dest], w)
    main_coefs[dest] = coef
    y_pred = X @ coef
    r2 = float(1 - np.var(Y[dest] - y_pred) / np.var(Y[dest])) if np.var(Y[dest]) > 0 else 0
    print(f"  {dest:10s}: R²={max(r2,0):.3f}")

# Bootstrap
print("[4/4] Bootstrap CIs...")
n = len(df)
boot = {dest: [] for dest in DEST_LABELS}
for b in range(N_BOOTSTRAP):
    idx = np.random.choice(n, n, replace=True)
    Xb = X[idx]; wb = w[idx]
    for dest in DEST_LABELS:
        boot[dest].append(wls(Xb, Y[dest][idx], wb))
    if (b+1) % 100 == 0: print(f"  Bootstrap: {b+1}/{N_BOOTSTRAP}")

# ── Extract betas for display parties ────────────────────────────────────────
# β_ij = coef[j] for party i = votes going to destination j per vote from party i in R1
# Normalize each row (source party) so betas sum to 1 across destinations
# — they should approximately already if model is well-specified

flows_out = []
for i, party in enumerate(DISPLAY_PARTIES):
    row = {'party': party, 'name': PARTY_NAMES[party], 'color': PARTY_COLORS[party]}
    idx_in_X = ALL_R1.index(party)

    raw = np.array([float(main_coefs[dest][idx_in_X]) for dest in DEST_LABELS])

    # Normalize: clip negative to 0, divide by sum
    raw_pos = np.clip(raw, 0, None)
    total = raw_pos.sum()
    flows = raw_pos / total if total > 1e-10 else np.ones(len(DEST_LABELS)) / len(DEST_LABELS)

    for j, label in enumerate(DEST_LABELS):
        row[f'to_{label}'] = round(float(flows[j]), 4)
        boot_vals = [float(boot[label][b][idx_in_X]) for b in range(N_BOOTSTRAP)]
        row[f'ci_{label}_lo'] = round(float(np.percentile(boot_vals, 5)), 4)
        row[f'ci_{label}_hi'] = round(float(np.percentile(boot_vals, 95)), 4)

    flows_out.append(row)
    vals = '  '.join(f"{l[0].upper()}={row[f'to_{l}']:.2f}" for l in DEST_LABELS)
    print(f"  {party:8s}: {vals}")

(DATA/'flows.json').write_text(json.dumps({
    "meta": {
        "timestamp":   datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "n_tables":    int(len(df)),
        "n_r1_regressors": len(ALL_R1),
        "method":      "OLS_vote_counts_no_intercept",
        "n_bootstrap": N_BOOTSTRAP,
        "note": "β_ij = votes to R2 destination j per vote from R1 party i",
    },
    "destinations": DEST_LABELS,
    "dest_names":   DEST_NAMES,
    "dest_colors":  DEST_COLORS,
    "parties":      flows_out,
}, ensure_ascii=False, indent=2))
print(f"\n✅  Saved: data/flows.json")
