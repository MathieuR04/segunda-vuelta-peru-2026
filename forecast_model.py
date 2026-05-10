"""
forecast_model.py — Segunda Vuelta Perú 2026
=============================================
Bottom-up table-level forecast.

SIGMA LOGIC (the key insight):
  At 7.6% reporting, 85,680 tables are uncounted.
  Even with perfect per-table predictions, errors accumulate.
  ±1 vote/table × 85,680 tables = ±85,680 vote floor.
  Realistic within-district prediction error is ~6 votes/table.
  sqrt(85680) × 6 = ±1,756 votes if all errors are INDEPENDENT.
  But errors are CORRELATED within districts/provinces/depts —
  all tables in an unreported district share the same prediction error.
  This correlation is what drives realistic uncertainty.

SIGMA CALIBRATION from 2021 within-group table-level std of Keiko share:
  Within polling place:  σ = 0.035 → ~4.6 votes/table
  Within district:       σ = 0.047 → ~6.2 votes/table
  Within province:       σ = 0.083 → ~10.8 votes/table
  Within department:     σ = 0.116 → ~15.1 votes/table
  National:              σ = 0.217 → ~28.1 votes/table

  R1 OLS reduces σ by sqrt(1-R²) at each level.
  Correlated component (all tables in same unobserved group share same error):
    Var_correlated = n_groups × (avg_tables_per_group × avg_vv × σ_group_mean)²
  where σ_group_mean = σ_within_group / sqrt(n_counted_in_group)

PREDICTION HIERARCHY for each uncounted table t:
  1. Same district has R2 data (live.json) → use district R2 mean + R1 correction
  2. Same province has R2 data → province R2 mean + R1 correction
  3. Same department has R2 data → dept R2 mean + R1 correction
  4. 2021 table-level result + national/dept swing → fallback
  5. National mean → last resort

NEVER use mesas_metadata.codigo_estado_acta for R2 counted/uncounted split.
All ubigeo joins: integer id_ubigeo (ONPE system), not district names.
"""

import json, datetime, warnings, unicodedata, math
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA   = Path('data')
N_SIMS = 10_000

# Calibrated σ within each geographic level from 2021 table-level data
# These are the FLOOR on prediction error — R1 OLS can only improve on them
SIGMA_WITHIN = {
    'district': 0.0474,
    'province': 0.0828,
    'dept':     0.1161,
    'national': 0.2165,
}
AVG_VV = 130   # avg valid votes per table in R2 (~electores * 0.75 * 0.93)

KEY = {'00000008':'fp','00000010':'jpp','00000035':'rla','00000016':'nieto',
       '00000014':'obras','00000023':'ppt','00000002':'an'}
FEATS = [f'{p}_sh' for p in KEY.values()]

def strip(s):
    if not isinstance(s, str): return ""
    return ''.join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != 'Mn').upper().strip()

def wls(X, y, w):
    Xw = X * np.sqrt(w)[:,None]
    yw = y * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return coef

def fit_flow_model(df_sub, feats=FEATS, y='r2_sh', min_n=5):
    """Fit weighted OLS. Returns (beta, r2) or (None, 0)."""
    d = df_sub.dropna(subset=[y] + feats)
    if len(d) < min_n:
        return None, 0.0
    X  = np.column_stack([np.ones(len(d))] + [d[f].values for f in feats])
    yv = d[y].values
    wv = d['avg_vv'].clip(lower=1).values
    beta = wls(X, yv, wv)
    resid = yv - X @ beta
    r2 = float(1 - np.var(resid) / np.var(yv)) if np.var(yv) > 0 else 0.0
    return beta, max(r2, 0.0)

def predict_r2(row, beta, feats=FEATS):
    x = np.array([1.0] + [float(row.get(f, 0)) for f in feats])
    return float(np.clip(x @ beta, 0.02, 0.98))

print("=" * 60)
print("FORECAST MODEL — Segunda Vuelta Perú 2026")
print("=" * 60)

# ── 1. Build id_ubigeo → shp_ubigeo bridge ────────────────────────────────────
print("\n[1/7] Ubigeo bridge...")
lu = pd.read_csv(DATA / 'ubigeo_lookup.csv')
for c in ['region','provincia','distrito']:
    lu[f'{c[0]}_s'] = lu[c].apply(strip)
gdf = gpd.read_file(DATA / 'dist.shp')
for col, field in [('DEPARTAMEN','r_s'),('PROVINCIA','p_s'),('DISTRITO','d_s')]:
    gdf[field] = gdf[col].apply(strip)
gdf['shp_ubigeo'] = gdf['UBIGEO'].astype(str).str.zfill(6)
lu2 = lu.rename(columns={'r_s':'r_s','p_s':'p_s','d_s':'d_s'})
lu2['r_s'] = lu['region'].apply(strip)
lu2['p_s'] = lu['provincia'].apply(strip)
lu2['d_s'] = lu['distrito'].apply(strip)
bridge = lu2.merge(gdf[['r_s','p_s','d_s','shp_ubigeo']],
    left_on=['r_s','p_s','d_s'], right_on=['r_s','p_s','d_s'], how='left'
)[['id_ubigeo','shp_ubigeo']].dropna()
id2shp = bridge.set_index('id_ubigeo')['shp_ubigeo'].to_dict()
print(f"  {len(bridge):,} ubigeo pairs bridged")

# ── 2. Load live.json ──────────────────────────────────────────────────────────
print("[2/7] Loading live.json...")
live   = json.loads((DATA / 'live.json').read_text())
lm     = live.get('meta', {})
n_cnt  = lm.get('counted_mesas', 0)
pct    = lm.get('pct_reported', 0.0)
k_live = lm.get('k_votes', 0)
s_live = lm.get('s_votes', 0)
vl     = lm.get('valid_votes', k_live + s_live) or 1
margin = k_live - s_live
lByU   = {d['u']: d for d in live.get('districts', [])}
print(f"  Counted: {n_cnt:,} ({pct:.1f}%)  Districts: {len(lByU):,}")
print(f"  Current margin: {margin:+,}")

# ── 3. Load metadata ───────────────────────────────────────────────────────────
print("[3/7] Loading metadata + R1 + 2021...")
meta = pd.read_parquet(DATA / 'mesas_metadata.parquet',
    columns=['codigo_mesa','id_ubigeo','eleccion','electores_habiles',
             'votos_validos','codigo_local_votacion'])
meta = meta[meta['eleccion'] == 'presidencial'].copy()
meta['id_ubigeo'] = meta['id_ubigeo'].fillna(0).astype(int)
meta['dept']      = (meta['id_ubigeo'] // 10000).astype(int)
meta['prov']      = (meta['id_ubigeo'] // 100).astype(int)
meta['dist']      = meta['id_ubigeo'].astype(int)
meta['place']     = meta['codigo_local_votacion'].fillna('UNK').astype(str)
meta['mesa_int']  = meta['codigo_mesa'].astype(int)
meta['shp_ubigeo']= meta['id_ubigeo'].map(id2shp)
meta['electores'] = meta['electores_habiles'].fillna(295).clip(lower=10)

# R1 party shares
res = pd.read_parquet(DATA / 'actas_resultados.parquet',
    columns=['codigo_mesa','eleccion','partido_codigo','votos'])
res = res[res['eleccion'] == 'presidencial'].copy()
res_k = res[res['partido_codigo'].isin(KEY)].copy()
res_k['feat'] = res_k['partido_codigo'].map(KEY)
r1w = res_k.pivot_table(index='codigo_mesa', columns='feat',
    values='votos', fill_value=0).reset_index()
r1w.columns.name = None
for p in KEY.values():
    if p not in r1w.columns: r1w[p] = 0
tvr1 = (res[~res['partido_codigo'].isin(['80','81','82'])]
    .groupby('codigo_mesa')['votos'].sum().reset_index(name='tv_r1'))
r1w = r1w.merge(tvr1, on='codigo_mesa', how='left')
r1w['tv_r1'] = r1w['tv_r1'].fillna(0)

# 2021 table-level (mesa code = MESA_DE_VOTACION integer)
df21 = pd.read_csv(DATA / 'Peruvian_Presidential_Election_Second_Round.csv',
    encoding='latin1', sep=';', index_col=False)
df21p = df21[df21['TIPO_ELECCION'] == 'PRESIDENCIAL'].copy()
df21p = df21p[df21p['UBIGEO'].astype(str).str.zfill(6).str[:2].astype(int) <= 25]
for c in ['VOTOS_P1','VOTOS_P2']:
    df21p[c] = pd.to_numeric(df21p[c], errors='coerce').fillna(0)
df21p['mesa_int'] = pd.to_numeric(df21p['MESA_DE_VOTACION'],
    errors='coerce').fillna(0).astype(int)
df21p['tv21'] = df21p['VOTOS_P1'] + df21p['VOTOS_P2']
df21p = df21p[df21p['tv21'] > 0].copy()
df21p['k21_sh'] = df21p['VOTOS_P2'] / df21p['tv21']
# District-level 2021
df21p['id_ubigeo'] = df21p['UBIGEO'].astype(str).str.zfill(6).astype(int)
d21_dist = df21p.groupby('id_ubigeo').agg(
    k21_sum=('VOTOS_P2','sum'), tv21_sum=('tv21','sum')).reset_index()
d21_dist['k21_sh_dist'] = d21_dist['k21_sum'] / d21_dist['tv21_sum']
k21_nat = float(df21p['VOTOS_P2'].sum() / df21p['tv21'].sum())
print(f"  2021 national Keiko share: {k21_nat:.4f}")

# ── 4. Build full feature matrix ───────────────────────────────────────────────
print("[4/7] Building features...")
df = (meta
    .merge(r1w, on='codigo_mesa', how='left')
    .merge(df21p[['mesa_int','k21_sh']], on='mesa_int', how='left')
    .merge(d21_dist[['id_ubigeo','k21_sh_dist']], on='id_ubigeo', how='left')
)
for p in KEY.values():
    if p not in df.columns: df[p] = 0
    df[p] = df[p].fillna(0)
df['tv_r1']   = df['tv_r1'].fillna(0)
df['k21_sh']  = df['k21_sh'].fillna(df['k21_sh_dist']).fillna(k21_nat)
df['avg_vv']  = df['tv_r1'].clip(lower=1) * 0.88   # imputed R2 valid votes
df['avg_vv']  = df['avg_vv'].clip(lower=50, upper=400)
for p in KEY.values():
    df[f'{p}_sh'] = np.where(df['tv_r1']>0, df[p]/df['tv_r1'], 0.0)

# Assign R2 share to each table from live.json district aggregate
df['live_d']   = df['shp_ubigeo'].map(lambda u: lByU.get(str(u)) if pd.notna(u) else None)
df['r2_sh']    = df['live_d'].map(
    lambda d: d['k']/(d['k']+d['s']) if d and (d['k']+d['s'])>0 else np.nan)
df['cm_frac']  = df['live_d'].map(lambda d: d['cm']/max(d['tm'],1) if d else 0.0)
df['cm_count'] = df['live_d'].map(lambda d: d['cm'] if d else 0)

# Counted/uncounted: for each district, mark exactly cm tables as "counted"
# (those with r2_sh available), rest as uncounted.
# This correctly handles districts with few counted tables.
df['is_cnt'] = False
for shp_ub, d_info in lByU.items():
    cm = d_info['cm']
    if cm <= 0: continue
    mask = df['shp_ubigeo'] == shp_ub
    idx = df[mask].index[:cm]  # take first cm tables from this district
    df.loc[idx, 'is_cnt'] = True

df_cnt = df[df['is_cnt']].copy()
df_unc = df[~df['is_cnt']].copy()
print(f"  Counted: {len(df_cnt):,}  Uncounted: {len(df_unc):,}")

# ── 5. Fit R1 vote-flow models at each geographic level ───────────────────────
print("[5/7] Fitting R1 vote-flow models at each level...")

# Counted tables: r2_sh = district aggregate (best proxy available without live_raw.parquet)
df_cnt_valid = df_cnt.dropna(subset=['r2_sh'])

# National model
beta_nat, r2_nat = fit_flow_model(df_cnt_valid, FEATS, min_n=20)
print(f"  National OLS: R²={r2_nat:.3f}  n={len(df_cnt_valid):,}")

# Department models
dept_models = {}
for dept_id, grp in df_cnt_valid.groupby('dept'):
    beta, r2 = fit_flow_model(grp, FEATS, min_n=5)
    dept_models[dept_id] = {'beta': beta, 'r2': r2, 'n': len(grp)}

# Province models
prov_models = {}
for prov_id, grp in df_cnt_valid.groupby('prov'):
    beta, r2 = fit_flow_model(grp, FEATS, min_n=3)
    prov_models[prov_id] = {'beta': beta, 'r2': r2, 'n': len(grp)}

# District models (few counted tables per district — mostly single values)
dist_models = {}
for dist_id, grp in df_cnt_valid.groupby('dist'):
    beta, r2 = fit_flow_model(grp, FEATS, min_n=3)
    dist_models[dist_id] = {
        'beta': beta, 'r2': r2, 'n': len(grp),
        'r2_mean': float(grp['r2_sh'].mean()),   # district R2 mean directly
        'cm': int(grp['cm_count'].iloc[0]) if len(grp) > 0 else 1,
    }

# R2 means at each level for uncounted tables
prov_r2_mean = df_cnt_valid.groupby('prov')['r2_sh'].agg(['mean','count','std'])
dept_r2_mean = df_cnt_valid.groupby('dept')['r2_sh'].agg(['mean','count','std'])
nat_r2_mean  = float(df_cnt_valid['r2_sh'].mean()) if len(df_cnt_valid) > 0 else k21_nat

# National swing: how much has Keiko shifted vs 2021 in counted areas?
# This is the key anchor — adjust 2021 prior by this swing for all predictions
k21_cnt_mean = float(df_cnt_valid['k21_sh'].mean()) if len(df_cnt_valid) > 0 else k21_nat
nat_swing = nat_r2_mean - k21_cnt_mean  # e.g. -0.03 means Keiko down 3pp vs 2021
# Shrink swing toward 0 when few tables counted (uncertain estimate)
n_cnt_tables = len(df_cnt_valid)
swing_confidence = min(1.0, n_cnt_tables / 5000.0)  # full confidence at 5000+ tables
nat_swing_adj = nat_swing * swing_confidence
print(f"  National swing vs 2021: {nat_swing:+.4f} (adj: {nat_swing_adj:+.4f}, confidence: {swing_confidence:.2f})")

print(f"  Dept models: {len(dept_models)}  Prov models: {len(prov_models)}  "
      f"Dist models: {len(dist_models)}")

# ── 6. Predict each uncounted table + assign σ ────────────────────────────────
print("[6/7] Predicting uncounted tables...")

mu_list    = []   # predicted Keiko share per uncounted table
sigma_list = []   # per-table σ IN VOTES
vv_list    = []   # imputed valid votes per table
group_list = []   # which geographic group (for correlated draws)
level_list = []   # which prediction level used

for _, row in df_unc.iterrows():
    dist_id = int(row['dist'])
    prov_id = int(row['prov'])
    dept_id = int(row['dept'])
    avg_vv_t = float(row['avg_vv'])

    # --- PREDICTION HIERARCHY ---
    # Level 1: district has R2 data — use district mean directly
    if dist_id in dist_models:
        dm     = dist_models[dist_id]
        cm     = dm['cm']
        mu_base = dm['r2_mean']
        if dm['beta'] is not None:
            mu_ols = predict_r2(row, dm['beta'])
            r2_d   = dm['r2']
            mu     = mu_base + (mu_ols - mu_base) * r2_d
        else:
            mu     = mu_base
            r2_d   = 0.0
        sigma_within = SIGMA_WITHIN['district'] * math.sqrt(1 - r2_d * 0.5)
        sigma_mean   = SIGMA_WITHIN['district'] / math.sqrt(max(cm, 1))
        sigma_sh     = math.sqrt(sigma_within**2 + sigma_mean**2)
        group_key    = ('dist', dist_id)
        level        = 'district'

    # Level 2: province has R2 data
    elif prov_id in prov_r2_mean.index:
        pr      = prov_r2_mean.loc[prov_id]
        mu_base = float(pr['mean'])
        if prov_id in prov_models and prov_models[prov_id]['beta'] is not None:
            pm     = prov_models[prov_id]
            mu_ols = predict_r2(row, pm['beta'])
            r2_p   = pm['r2']
            mu     = mu_base + (mu_ols - mu_base) * r2_p
        else:
            mu     = mu_base
            r2_p   = 0.0
        sigma_sh  = SIGMA_WITHIN['province'] * math.sqrt(1 - r2_p * 0.5)
        group_key = ('prov', prov_id)
        level     = 'province'

    # Level 3: department has R2 data
    elif dept_id in dept_r2_mean.index:
        dr     = dept_r2_mean.loc[dept_id]
        mu_base = float(dr['mean'])
        if dept_id in dept_models and dept_models[dept_id]['beta'] is not None:
            dm2    = dept_models[dept_id]
            mu_ols = predict_r2(row, dm2['beta'])
            r2_k   = dm2['r2']
            mu     = mu_base + (mu_ols - mu_base) * r2_k
        else:
            mu     = mu_base
            r2_k   = 0.0
        sigma_sh  = SIGMA_WITHIN['dept'] * math.sqrt(1 - r2_k * 0.5)
        group_key = ('dept', dept_id)
        level     = 'dept'

    # Level 4: R1 national model + 2021 anchor
    else:
        k21_t = float(np.clip(row.get('k21_sh', k21_nat) + nat_swing_adj, 0.02, 0.98))
        if beta_nat is not None:
            mu_ols = predict_r2(row, beta_nat)
            # Blend 2021 prior and OLS (2021 weighted by TAU, OLS by R² strength)
            mu = mu_ols * r2_nat + k21_t * (1 - r2_nat)
        else:
            mu = k21_t
        sigma_sh  = SIGMA_WITHIN['national'] * math.sqrt(1 - r2_nat * 0.3)
        group_key = ('nat', 0)
        level     = 'national'

    mu = float(np.clip(mu, 0.02, 0.98))

    # σ in votes for this table
    sigma_votes = sigma_sh * avg_vv_t

    mu_list.append(mu)
    sigma_list.append(sigma_votes)
    vv_list.append(avg_vv_t)
    group_list.append(group_key)
    level_list.append(level)

mu_arr    = np.array(mu_list)
sigma_arr = np.array(sigma_list)
vv_arr    = np.array(vv_list)

level_counts = pd.Series(level_list).value_counts()
print(f"  Prediction levels:")
for lv, cnt in level_counts.items():
    med_sig = np.median([sigma_list[i] for i,l in enumerate(level_list) if l==lv])
    print(f"    {lv:12s}: {cnt:6,} tables  median σ={med_sig:.1f} votes/table")

# ── 7. Monte Carlo with correct correlated error structure ────────────────────
# KEY: errors are correlated within each (level, group_id) cluster.
# All uncounted tables in the same district share the same district-level shock.
# All tables in the same province share the same province-level shock.
# These do NOT cancel — they accumulate linearly for each cluster.
#
# Total variance = Σ_t σ²_within(t)            [independent, cancels as sqrt(n)]
#               + Σ_groups (Σ_t∈group vv_t)² × σ²_group_mean
#                 [correlated, grows as n]
#
# We implement this directly in the MC by drawing one shared error per group.

print(f"\n[7/7] Monte Carlo ({N_SIMS:,} simulations)...")

# Unique groups
unique_groups = list(set(group_list))
group_arr = np.array(group_list, dtype=object)  # shape (n_tables,) of tuples

# For each group: compute the GROUP-LEVEL σ (uncertainty in the group mean prediction)
# This is what creates correlated errors across all tables in the group.
group_sigma = {}
for gk in unique_groups:
    mask = group_arr == gk
    gtype, gid = gk
    if gtype == 'dist':
        dm = dist_models.get(gid, {})
        cm = dm.get('cm', 1)
        r2 = dm.get('r2', 0.0)
        # Uncertainty in district mean = within_dist_σ / sqrt(cm_counted)
        # Plus: are the counted tables representative of uncounted ones? → add floor
        sigma_g = SIGMA_WITHIN['district'] / math.sqrt(max(cm, 1))
        sigma_g = max(sigma_g, SIGMA_WITHIN['district'] * 0.3)   # floor
    elif gtype == 'prov':
        pr = prov_r2_mean.loc[gid] if gid in prov_r2_mean.index else None
        n  = int(pr['count']) if pr is not None else 1
        sigma_g = SIGMA_WITHIN['province'] / math.sqrt(max(n, 1))
        sigma_g = max(sigma_g, SIGMA_WITHIN['province'] * 0.4)
    elif gtype == 'dept':
        dr = dept_r2_mean.loc[gid] if gid in dept_r2_mean.index else None
        n  = int(dr['count']) if dr is not None else 1
        sigma_g = SIGMA_WITHIN['dept'] / math.sqrt(max(n, 1))
        sigma_g = max(sigma_g, SIGMA_WITHIN['dept'] * 0.5)
    else:  # national
        sigma_g = SIGMA_WITHIN['national']
    group_sigma[gk] = sigma_g

# Group total vote weight (for computing correlated variance contribution)
group_to_idx = {gk: np.array([i for i, g in enumerate(group_list) if g == gk]) for gk in unique_groups}

group_vv_sum = {}
for gk in unique_groups:
    idxs = group_to_idx[gk]
    group_vv_sum[gk] = float(vv_arr[idxs].sum())

# Print uncertainty breakdown
print(f"  Uncertainty components:")
var_independent = float(np.sum(sigma_arr**2))
var_correlated  = sum(
    group_vv_sum[gk]**2 * group_sigma[gk]**2
    for gk in unique_groups
)
sigma_ind  = math.sqrt(var_independent)
sigma_corr = math.sqrt(var_correlated)
sigma_tot_approx = math.sqrt(var_independent + var_correlated)
print(f"    Independent (within-group): ±{sigma_ind:,.0f} votes")
print(f"    Correlated (group mean error): ±{sigma_corr:,.0f} votes")
print(f"    Total (approx):  ±{sigma_tot_approx:,.0f} votes")
print(f"    Expected 95% CI: ±{1.96*sigma_tot_approx:,.0f} votes")

# MC simulation
all_margins = np.zeros(N_SIMS)
for sim in range(N_SIMS):
    # For each table: draw a share-space prediction
    # = mu(t) + group_shock(share) + table_noise(share)
    # Then convert to votes

    # Table-level independent noise (in share space)
    table_sigma_sh = sigma_arr / vv_arr   # convert votes back to share
    table_noise_sh = np.random.normal(0, table_sigma_sh)

    # Group-level correlated shock (in share space, shared by all tables in group)
    keiko_sh = mu_arr + table_noise_sh
    for gk, idxs in group_to_idx.items():
        group_shock_sh = np.random.normal(0, group_sigma[gk])
        keiko_sh[idxs] += group_shock_sh

    keiko_sh = np.clip(keiko_sh, 0.0, 1.0)
    k_unc = float((keiko_sh * vv_arr).sum())
    s_unc = float(((1 - keiko_sh) * vv_arr).sum())
    all_margins[sim] = (k_live + k_unc) - (s_live + s_unc)

win_k   = float((all_margins > 0).mean())
proj_m  = float(np.median(all_margins))
ci95_lo = float(np.percentile(all_margins, 2.5))
ci95_hi = float(np.percentile(all_margins, 97.5))
ci80_lo = float(np.percentile(all_margins, 10))
ci80_hi = float(np.percentile(all_margins, 90))
sigma_t = float(np.std(all_margins))
proj_k_pct = (k_live + float(np.sum(mu_arr * vv_arr))) / (vl + float(vv_arr.sum())) * 100

print(f"\n{'='*50}")
print(f"FORECAST RESULTS")
print(f"{'='*50}")
print(f"  Win prob Keiko:   {win_k:.1%}")
print(f"  Win prob Sánchez: {1-win_k:.1%}")
print(f"  Current margin:   {margin:+,}")
print(f"  Projected margin: {proj_m:+,.0f}")
print(f"  95% CI:  [{ci95_lo:+,.0f}, {ci95_hi:+,.0f}]")
print(f"  80% CI:  [{ci80_lo:+,.0f}, {ci80_hi:+,.0f}]")
print(f"  σ total: {sigma_t:,.0f} votes")

hist_v, hist_e = np.histogram(all_margins, bins=80)
out = DATA / 'forecast.json'
out.write_text(json.dumps({
    "meta": {
        "timestamp":        datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "counted_mesas":    n_cnt,
        "pct_reported":     round(pct, 2),
        "model_r2_nat":     round(r2_nat, 4),
        "sigma_total":      int(sigma_t),
        "sigma_independent":int(sigma_ind),
        "sigma_correlated": int(sigma_corr),
        "n_sims":           N_SIMS,
        "status":           "alta" if pct>50 else "moderada" if pct>10 else "baja",
    },
    "results": {
        "win_prob_keiko":   round(win_k, 4),
        "win_prob_sanchez": round(1 - win_k, 4),
        "current_margin":   int(margin),
        "proj_margin":      int(proj_m),
        "proj_k_pct":       round(proj_k_pct, 3),
        "proj_s_pct":       round(100 - proj_k_pct, 3),
        "ci_95_lo":         int(ci95_lo),
        "ci_95_hi":         int(ci95_hi),
        "ci_80_lo":         int(ci80_lo),
        "ci_80_hi":         int(ci80_hi),
        "sigma":            int(sigma_t),
    },
    "distribution": {
        "bins":   [int((hist_e[i]+hist_e[i+1])/2) for i in range(len(hist_v))],
        "counts": [int(x) for x in hist_v],
    }
}, ensure_ascii=False, indent=2))
print(f"\n✅  Saved: {out}")
