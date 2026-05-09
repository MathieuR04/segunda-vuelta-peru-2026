"""
forecast_model.py
=================
Hierarchical Bayesian forecast for the 2026 Peruvian presidential runoff.

DATA INPUTS (all at polling-table level where possible):
  data/live_raw.parquet          — R2 2026 live results (one row per counted table)
  data/mesas_metadata.parquet    — all 92,766 tables with ubigeo hierarchy
  data/actas_resultados.parquet  — R1 2026 results per table × party
  data/Peruvian_Presidential_Election_Second_Round.csv — R2 2021 table-level

OUTPUT:
  data/forecast.json

MODEL ARCHITECTURE:
  For each uncounted table t in district d, province p, department k:

  Step 1 — Build feature vector from R1 2026 (table-level):
    X(t) = [fp_sh, jpp_sh, rla_sh, nieto_sh, obras_sh, ppt_sh, an_sh,
             blancos_r1_sh, nulos_r1_sh, keiko_21_sh (if available)]

  Step 2 — Fit national OLS on counted tables:
    keiko_r2_sh(t) ~ X(t)·β_nat + ε_nat

  Step 3 — Compute within-district residuals for counted tables.
    For each geographic level (dist, prov, dept), compute:
      mean_residual_level: average systematic deviation from national model
      n_level:             number of counted tables at that level
      σ_level:             residual std at that level

  Step 4 — Bayesian prediction for each uncounted table via precision weighting:
    The prediction at each level is:
      μ_dist = national_pred + dist_residual   (if n_dist >= MIN_DIST_N)
      μ_prov = national_pred + prov_residual   (if n_prov >= MIN_PROV_N)
      μ_dept = national_pred + dept_residual

    Combine via precision weighting (inverse variance):
      precision_dist = n_dist / σ²_dist (if available, else 0)
      precision_prov = n_prov / σ²_prov
      precision_dept = n_dept / σ²_dept
      precision_nat  = n_national / σ²_nat / K  (K downweights national)
      precision_2021 = τ_2021 / σ²_2021  (Bayesian prior from 2021)

      μ_final = weighted_mean(μ_dist, μ_prov, μ_dept, μ_nat, μ_2021,
                              weights=[prec_dist, prec_prov, prec_dept, prec_nat, prec_2021])

  Step 5 — Uncertainty for each uncounted table:
      σ²_pred(t) = 1 / Σ(precisions)        # posterior variance
                 + σ²_residual_nat           # irreducible model error
    This σ grows when: few local counted tables, district differs from province,
    geography not covered in 2021 data.

  Step 6 — Monte Carlo (N_SIMS=10,000):
    For each simulation:
      - Draw keiko_r2_sh(t) ~ N(μ_final(t), σ²_pred(t)) for all uncounted t
      - Clip to [0,1], multiply by imputed valid votes → votes
      - Sum with current counted votes → simulated national margin
    → Distribution of 10,000 final margins → 95% CI, win probabilities

UBIGEO HIERARCHY (using integer codes, not names — avoids duplicate district names):
  Department: id_ubigeo // 10000        (e.g. 15 = Lima)
  Province:   id_ubigeo // 100          (e.g. 1501 = Lima province)
  District:   id_ubigeo                 (e.g. 150101 = Lima district)

REQUIREMENTS:
  pip install pandas pyarrow numpy scipy
"""

import json, datetime, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

DATA   = Path('data')
N_SIMS = 10_000
SEED   = 42
np.random.seed(SEED)

# Bayesian prior strength from 2021 (equivalent number of virtual tables)
TAU_2021 = 8.0
# Minimum tables in a district to trust district-level residual
MIN_DIST_N = 3
MIN_PROV_N = 5

# Party codes → feature names
KEY_PARTIES = {
    '00000008': 'fp',
    '00000010': 'jpp',
    '00000035': 'rla',
    '00000016': 'nieto',
    '00000014': 'obras',
    '00000023': 'ppt',
    '00000002': 'an',
    '80':        'bl_r1',
    '81':        'nu_r1',
}
FEAT_COLS = ['fp','jpp','rla','nieto','obras','ppt','an','bl_r1','nu_r1']

print("=" * 60)
print("FORECAST MODEL — Segunda Vuelta Perú 2026")
print("=" * 60)

# ── STEP 0: Load metadata ──────────────────────────────────────────────────────
print("\n[1/6] Loading metadata...")
meta = pd.read_parquet(
    DATA / 'mesas_metadata.parquet',
    columns=['codigo_mesa','id_ubigeo','eleccion','codigo_estado_acta',
             'codigo_local_votacion','electores_habiles','votos_validos']
)
meta = meta[meta['eleccion'] == 'presidencial'].copy()
meta['id_ubigeo'] = meta['id_ubigeo'].fillna(0).astype(int)
meta['dept']   = (meta['id_ubigeo'] // 10000).astype(int)
meta['prov']   = (meta['id_ubigeo'] // 100).astype(int)
meta['dist']   = meta['id_ubigeo'].astype(int)
meta['place']  = meta['codigo_local_votacion'].fillna('UNK')
meta['mesa_int'] = meta['codigo_mesa'].astype(int)
print(f"  Tables: {len(meta):,}")

# ── STEP 1: Load R1 2026 (table level) ────────────────────────────────────────
print("[2/6] Loading R1 2026 results...")
res_r1 = pd.read_parquet(
    DATA / 'actas_resultados.parquet',
    columns=['codigo_mesa','eleccion','partido_codigo','votos']
)
res_r1 = res_r1[res_r1['eleccion'] == 'presidencial'].copy()
res_r1_key = res_r1[res_r1['partido_codigo'].isin(KEY_PARTIES)].copy()
res_r1_key['feat'] = res_r1_key['partido_codigo'].map(KEY_PARTIES)
r1_wide = (res_r1_key
    .pivot_table(index='codigo_mesa', columns='feat', values='votos', fill_value=0)
    .reset_index())
r1_wide.columns.name = None
for c in FEAT_COLS:
    if c not in r1_wide.columns:
        r1_wide[c] = 0

# Total valid (excl blancos/nulos/impugnados) and total emitidos per table
total_valid_r1 = (res_r1[~res_r1['partido_codigo'].isin(['80','81','82'])]
    .groupby('codigo_mesa')['votos'].sum().reset_index(name='tv_r1'))
total_all_r1 = (res_r1.groupby('codigo_mesa')['votos']
    .sum().reset_index(name='ta_r1'))

r1_wide = r1_wide.merge(total_valid_r1, on='codigo_mesa', how='left')
r1_wide = r1_wide.merge(total_all_r1,   on='codigo_mesa', how='left')
r1_wide['tv_r1'] = r1_wide['tv_r1'].fillna(0)
r1_wide['ta_r1'] = r1_wide['ta_r1'].fillna(0)
print(f"  R1 tables: {len(r1_wide):,}")

# ── STEP 2: Load R2 2021 (table level, 93% coverage) ─────────────────────────
print("[3/6] Loading R2 2021 (table-level anchor)...")
df21 = pd.read_csv(
    DATA / 'Peruvian_Presidential_Election_Second_Round.csv',
    encoding='latin1', sep=';', index_col=False
)
df21_pres = df21[df21['TIPO_ELECCION'] == 'PRESIDENCIAL'].copy()
for col in ['VOTOS_P1','VOTOS_P2','VOTOS_VB','VOTOS_VN']:
    df21_pres[col] = pd.to_numeric(df21_pres[col], errors='coerce').fillna(0)
df21_pres = df21_pres.rename(columns={
    'MESA_DE_VOTACION': 'mesa_int',
    'VOTOS_P1': 'castillo_21',
    'VOTOS_P2': 'keiko_21',
    'VOTOS_VB': 'blancos_21',
    'VOTOS_VN': 'nulos_21',
})
df21_pres['tv_21'] = df21_pres['castillo_21'] + df21_pres['keiko_21']
df21_pres = df21_pres[df21_pres['tv_21'] > 0][['mesa_int','castillo_21','keiko_21','blancos_21','nulos_21','tv_21']]
print(f"  2021 tables: {len(df21_pres):,}")

# ── STEP 3: Load R2 2026 live results (table level) ───────────────────────────
print("[4/6] Loading R2 2026 live results...")
live_json = json.loads((DATA / 'live.json').read_text())
live_meta = live_json.get('meta', {})
counted_mesas_set = set()

# Try table-level parquet first
live_raw_path = DATA / 'live_raw.parquet'
if live_raw_path.exists():
    live_raw = pd.read_parquet(live_raw_path)
    live_raw = live_raw[live_raw['estado'] == 'C'].copy()
    live_raw['k_votes'] = pd.to_numeric(live_raw['k_votes'], errors='coerce').fillna(0)
    live_raw['s_votes'] = pd.to_numeric(live_raw['s_votes'], errors='coerce').fillna(0)
    live_raw['blancos'] = pd.to_numeric(live_raw['blancos'], errors='coerce').fillna(0)
    live_raw['nulos']   = pd.to_numeric(live_raw['nulos'],   errors='coerce').fillna(0)
    live_raw['tv_r2']   = live_raw['k_votes'] + live_raw['s_votes']
    live_raw['keiko_r2_sh'] = np.where(
        live_raw['tv_r2'] > 0, live_raw['k_votes'] / live_raw['tv_r2'], np.nan)
    counted_mesas_set = set(live_raw['codigo_mesa'])
    print(f"  Live tables (parquet): {len(live_raw):,}")
    use_table_level_r2 = True
else:
    # Fall back to district-level aggregates from live.json
    print("  WARNING: live_raw.parquet not found. Using district aggregates.")
    print("  Forecast uncertainty will be larger.")
    live_raw = pd.DataFrame()
    use_table_level_r2 = False

n_counted = live_meta.get('counted_mesas', 0)
n_uncounted = 92766 - n_counted
pct_rep = live_meta.get('pct_reported', 0)
k_total_live = live_meta.get('k_votes', 0)
s_total_live = live_meta.get('s_votes', 0)
current_margin = k_total_live - s_total_live
print(f"  Counted: {n_counted:,} ({pct_rep:.1f}%)")
print(f"  Current margin: {current_margin:+,} (Keiko {'ahead' if current_margin>0 else 'behind'})")

# ── STEP 4: Build full feature matrix ─────────────────────────────────────────
print("[5/6] Building feature matrix and fitting model...")

df = (meta
    .merge(r1_wide, on='codigo_mesa', how='left')
    .merge(df21_pres, on='mesa_int', how='left')
)
for c in FEAT_COLS:
    df[c] = df[c].fillna(0)
df['tv_r1']   = df['tv_r1'].fillna(0)
df['ta_r1']   = df['ta_r1'].fillna(0)
df['tv_21']   = df['tv_21'].fillna(0)
df['keiko_21_sh'] = np.where(df['tv_21'] > 0, df['keiko_21'] / df['tv_21'], np.nan)
df['castillo_21_sh'] = np.where(df['tv_21'] > 0, df['castillo_21'] / df['tv_21'], np.nan)

# Normalize R1 shares by total valid votes
for p in FEAT_COLS:
    df[f'{p}_sh'] = np.where(df['tv_r1'] > 0, df[p] / df['tv_r1'], 0.0)

FEAT_SH = [f'{p}_sh' for p in FEAT_COLS] + ['keiko_21_sh']

# Split counted vs uncounted
if use_table_level_r2 and len(live_raw) > 0:
    df_counted   = df[df['codigo_mesa'].isin(counted_mesas_set)].copy()
    df_uncounted = df[~df['codigo_mesa'].isin(counted_mesas_set)].copy()
    
    # Merge R2 results onto counted tables
    df_counted = df_counted.merge(
        live_raw[['codigo_mesa','k_votes','s_votes','blancos','nulos','tv_r2','keiko_r2_sh']],
        on='codigo_mesa', how='left'
    )
    df_counted = df_counted[df_counted['keiko_r2_sh'].notna()].copy()
else:
    # No table-level R2 — use district aggregates as proxy
    df_counted   = df[df['codigo_estado_acta'] == 'C'].copy()
    df_uncounted = df[df['codigo_estado_acta'] != 'C'].copy()
    # Assign R2 share from live.json district aggregates
    lByU = {d['u']: d for d in live_json.get('districts', [])}
    df_counted['keiko_r2_sh'] = df_counted['dist'].map(
        lambda u: lByU[str(u)]['k']/(lByU[str(u)]['k']+lByU[str(u)]['s'])
                  if str(u) in lByU and (lByU[str(u)]['k']+lByU[str(u)]['s'])>0 else np.nan
    )
    df_counted = df_counted[df_counted['keiko_r2_sh'].notna()]
    df_counted['tv_r2'] = df_counted['votos_validos'].fillna(df_counted['tv_r1'] * 0.92)
    df_counted['k_votes'] = df_counted['keiko_r2_sh'] * df_counted['tv_r2']
    df_counted['s_votes'] = (1 - df_counted['keiko_r2_sh']) * df_counted['tv_r2']

print(f"  Counted tables for fitting: {len(df_counted):,}")
print(f"  Uncounted tables to predict: {len(df_uncounted):,}")

if len(df_counted) < 10:
    print("  Too few counted tables — using 2021-only prior")
    beta_nat = None
    sigma_nat = 0.12
    national_mean_pred = 0.5
else:
    # ── National OLS (with 2021 share as feature) ──────────────────────────────
    # Features: R1 party shares + 2021 keiko share
    X_cols = [f'{p}_sh' for p in FEAT_COLS] + ['keiko_21_sh']
    df_fit = df_counted.dropna(subset=['keiko_r2_sh']).copy()
    # Fill missing 2021 with national mean
    nat_keiko_21_sh = df_fit['keiko_21_sh'].mean()
    df_fit['keiko_21_sh'] = df_fit['keiko_21_sh'].fillna(nat_keiko_21_sh)
    
    X = df_fit[X_cols].fillna(0).values
    y = df_fit['keiko_r2_sh'].values
    w = np.sqrt(df_fit['tv_r2'].clip(lower=1).values)  # weight by sqrt(valid votes)
    
    # Weighted OLS via normal equations
    X_aug = np.column_stack([np.ones(len(X)), X])
    W = np.diag(w)
    try:
        XtWX = X_aug.T @ W @ X_aug
        XtWy = X_aug.T @ W @ y
        beta_nat = np.linalg.solve(XtWX + np.eye(len(XtWX))*1e-8, XtWy)
        y_pred_nat = X_aug @ beta_nat
        residuals_nat = y - y_pred_nat
        sigma_nat = np.std(residuals_nat)
        r2_nat = 1 - np.var(residuals_nat)/np.var(y)
        national_mean_pred = np.mean(y_pred_nat)
        print(f"  National OLS R²: {r2_nat:.4f}  σ: {sigma_nat:.4f}  n={len(df_fit):,}")
    except:
        beta_nat = None
        sigma_nat = 0.12
        national_mean_pred = df_counted['keiko_r2_sh'].mean() if len(df_counted)>0 else 0.5
        print(f"  OLS failed — using national mean {national_mean_pred:.4f}")

# ── Compute geographic residuals at district/province/dept level ───────────────
def geo_residuals(df_fit_with_residuals, level_col, min_n):
    """Compute mean residual and σ at each geographic level."""
    grp = df_fit_with_residuals.groupby(level_col)['residual']
    stats_df = pd.DataFrame({
        'mean_resid': grp.mean(),
        'std_resid':  grp.std().fillna(0.08),
        'n':          grp.count(),
    }).reset_index()
    stats_df['std_resid'] = stats_df['std_resid'].clip(lower=0.02)
    # Only trust levels with enough tables
    stats_df.loc[stats_df['n'] < min_n, 'mean_resid'] = 0.0
    return stats_df

if beta_nat is not None and len(df_counted) >= 10:
    df_fit2 = df_counted.dropna(subset=['keiko_r2_sh']).copy()
    df_fit2['keiko_21_sh'] = df_fit2['keiko_21_sh'].fillna(nat_keiko_21_sh)
    X2 = df_fit2[[f'{p}_sh' for p in FEAT_COLS]+['keiko_21_sh']].fillna(0).values
    X2_aug = np.column_stack([np.ones(len(X2)), X2])
    df_fit2['nat_pred'] = X2_aug @ beta_nat
    df_fit2['residual'] = df_fit2['keiko_r2_sh'] - df_fit2['nat_pred']
    
    resid_dist  = geo_residuals(df_fit2, 'dist',  MIN_DIST_N)
    resid_prov  = geo_residuals(df_fit2, 'prov',  MIN_PROV_N)
    resid_dept  = geo_residuals(df_fit2, 'dept',  2)
    
    # Index by geo code
    rd_dict = resid_dist.set_index('dist').to_dict('index')
    rp_dict = resid_prov.set_index('prov').to_dict('index')
    rk_dict = resid_dept.set_index('dept').to_dict('index')
    
    print(f"  Districts with residual data: {len(resid_dist[resid_dist['n']>=MIN_DIST_N]):,}")
    print(f"  Provinces with residual data: {len(resid_prov[resid_prov['n']>=MIN_PROV_N]):,}")
    print(f"  Departments with residual data: {len(resid_dept):,}")
else:
    rd_dict = rp_dict = rk_dict = {}

# ── STEP 5: Predict each uncounted table ──────────────────────────────────────
# Impute valid votes for uncounted tables
# Use R2 2021 turnout if available, else R1 total with correction factor
if len(df_counted) > 0:
    tv_correction = (df_counted['tv_r2'].mean() / 
                    df_counted['tv_r1'].clip(lower=1).mean()) if len(df_counted)>0 else 0.88
else:
    tv_correction = 0.88  # R2 typically ~12% lower than R1 valid votes
tv_correction = np.clip(tv_correction, 0.75, 1.05)
print(f"  Turnout correction (R2/R1): {tv_correction:.3f}")

nat_keiko_21_sh_global = df['keiko_21_sh'].mean() if df['keiko_21_sh'].notna().any() else 0.5

# National σ from 2021 (across all tables, how much did keiko_21 vary from district mean?)
sigma_2021 = df['keiko_21_sh'].std() if df['keiko_21_sh'].notna().any() else 0.12
sigma_2021 = max(sigma_2021, 0.06)

mu_list    = []  # predicted keiko R2 share per uncounted table
sigma_list = []  # uncertainty per uncounted table
tv_list    = []  # imputed valid votes

for _, row in df_uncounted.iterrows():
    dist_id = int(row['dist'])
    prov_id = int(row['prov'])
    dept_id = int(row['dept'])
    
    # National prediction
    if beta_nat is not None:
        x_feat = np.array([1.0] + 
            [row.get(f'{p}_sh', 0) for p in FEAT_COLS] +
            [row['keiko_21_sh'] if pd.notna(row['keiko_21_sh']) else nat_keiko_21_sh_global])
        mu_nat = float(x_feat @ beta_nat)
        mu_nat = np.clip(mu_nat, 0.02, 0.98)
    else:
        mu_nat = national_mean_pred if national_mean_pred is not None else 0.5
    
    # 2021 table-level prior
    has_2021 = pd.notna(row.get('keiko_21_sh'))
    if has_2021:
        mu_2021 = float(row['keiko_21_sh'])
        # Adjust 2021 by the national swing observed in counted tables
        if len(df_counted) > 0 and beta_nat is not None:
            swing = national_mean_pred - (df['keiko_21_sh'].mean() if df['keiko_21_sh'].notna().any() else 0.5)
            mu_2021 = np.clip(mu_2021 + swing * 0.6, 0.02, 0.98)
        prec_2021 = TAU_2021 / (sigma_2021 ** 2)
    else:
        mu_2021 = mu_nat
        prec_2021 = TAU_2021 / 4 / (sigma_2021 ** 2)  # weaker prior
    
    # Geographic residual adjustments with precisions
    prec_nat  = max(0, len(df_counted)) / max(sigma_nat**2, 0.001) / 50.0
    
    # District level
    rd = rd_dict.get(dist_id, {})
    if rd and rd.get('n', 0) >= MIN_DIST_N:
        mu_dist   = mu_nat + rd['mean_resid']
        prec_dist = rd['n'] / max(rd['std_resid']**2, 0.001)
    else:
        mu_dist   = mu_nat
        prec_dist = 0.0
    
    # Province level
    rp = rp_dict.get(prov_id, {})
    if rp and rp.get('n', 0) >= MIN_PROV_N:
        mu_prov   = mu_nat + rp['mean_resid']
        prec_prov = rp['n'] / max(rp['std_resid']**2, 0.001) / 3.0
    else:
        mu_prov   = mu_nat
        prec_prov = 0.0
    
    # Department level
    rk = rk_dict.get(dept_id, {})
    if rk:
        mu_dept   = mu_nat + rk.get('mean_resid', 0)
        prec_dept = max(rk.get('n',0),1) / max(rk.get('std_resid',0.08)**2, 0.001) / 8.0
    else:
        mu_dept   = mu_nat
        prec_dept = 0.0
    
    # Precision-weighted combination
    total_prec = prec_2021 + prec_dist + prec_prov + prec_dept + prec_nat
    total_prec = max(total_prec, 1e-6)
    
    mu_final = (prec_2021*mu_2021 + prec_dist*mu_dist + 
                prec_prov*mu_prov + prec_dept*mu_dept + 
                prec_nat*mu_nat) / total_prec
    mu_final = np.clip(mu_final, 0.02, 0.98)
    
    # Posterior variance: 1/total_precision + irreducible model error
    sigma_pred = np.sqrt(1.0/total_prec + sigma_nat**2)
    sigma_pred = np.clip(sigma_pred, 0.02, 0.25)
    
    # Imputed valid votes
    tv_r1_row = row.get('tv_r1', 0) or 0
    tv_21_row = row.get('tv_21', 0) or 0
    if tv_r1_row > 0:
        tv_imputed = tv_r1_row * tv_correction
    elif tv_21_row > 0:
        tv_imputed = tv_21_row * (tv_correction * 1.05)
    else:
        tv_imputed = 150.0  # national average valid votes per table
    
    mu_list.append(mu_final)
    sigma_list.append(sigma_pred)
    tv_list.append(tv_imputed)

mu_arr    = np.array(mu_list)
sigma_arr = np.array(sigma_list)
tv_arr    = np.array(tv_list)

print(f"  Predicted {len(mu_arr):,} uncounted tables")
print(f"  Mean predicted Keiko share in uncounted: {mu_arr.mean():.3f}")
print(f"  Mean σ per table: {sigma_arr.mean():.4f}")

# ── STEP 6: Monte Carlo simulation ────────────────────────────────────────────
print(f"\n[6/6] Monte Carlo simulation ({N_SIMS:,} runs)...")

# Draw from N(mu, sigma) for each uncounted table × each simulation
# Shape: (n_uncounted, N_SIMS)
draws = np.random.normal(
    mu_arr[:, None],
    sigma_arr[:, None],
    size=(len(mu_arr), N_SIMS)
)
draws = np.clip(draws, 0.0, 1.0)

# Votes per table per simulation
k_draws = draws       * tv_arr[:, None]
s_draws = (1 - draws) * tv_arr[:, None]

# National totals per simulation
k_uncounted_sims = k_draws.sum(axis=0)
s_uncounted_sims = s_draws.sum(axis=0)

# Add current counted votes
k_final_sims = k_total_live + k_uncounted_sims
s_final_sims = s_total_live + s_uncounted_sims
margin_sims  = k_final_sims - s_final_sims

# Results
win_prob_keiko   = float((margin_sims > 0).mean())
win_prob_sanchez = 1.0 - win_prob_keiko
proj_margin      = float(np.median(margin_sims))
ci_lo_95         = float(np.percentile(margin_sims, 2.5))
ci_hi_95         = float(np.percentile(margin_sims, 97.5))
ci_lo_80         = float(np.percentile(margin_sims, 10))
ci_hi_80         = float(np.percentile(margin_sims, 90))
sigma_total      = float(np.std(margin_sims))

# Expected final vote shares
k_pct_sims = k_final_sims / (k_final_sims + s_final_sims) * 100
proj_k_pct = float(np.median(k_pct_sims))
ci_k_lo    = float(np.percentile(k_pct_sims, 2.5))
ci_k_hi    = float(np.percentile(k_pct_sims, 97.5))

print(f"\n{'='*50}")
print(f"FORECAST RESULTS")
print(f"{'='*50}")
print(f"  Win probability — Keiko:   {win_prob_keiko:.1%}")
print(f"  Win probability — Sánchez: {win_prob_sanchez:.1%}")
print(f"  Projected final margin: {proj_margin:+,.0f} votes")
print(f"  95% CI: [{ci_lo_95:+,.0f}, {ci_hi_95:+,.0f}]")
print(f"  Projected Keiko share: {proj_k_pct:.2f}% [{ci_k_lo:.2f}%, {ci_k_hi:.2f}%]")
print(f"  σ (total): {sigma_total:,.0f} votes")

# ── Output ────────────────────────────────────────────────────────────────────
# Export histogram bins for the NYT-style needle distribution
hist_vals, hist_edges = np.histogram(margin_sims, bins=80)
hist_centers = ((hist_edges[:-1] + hist_edges[1:]) / 2).tolist()

forecast = {
    "meta": {
        "timestamp":       datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "counted_mesas":   n_counted,
        "pct_reported":    round(pct_rep, 2),
        "use_table_level": use_table_level_r2,
        "model_r2":        round(r2_nat if beta_nat is not None and 'r2_nat' in dir() else 0, 4),
        "sigma_nat":       round(sigma_nat, 4),
        "n_sims":          N_SIMS,
    },
    "results": {
        "win_prob_keiko":   round(win_prob_keiko, 4),
        "win_prob_sanchez": round(win_prob_sanchez, 4),
        "current_margin":   int(current_margin),
        "proj_margin":      int(proj_margin),
        "proj_k_pct":       round(proj_k_pct, 3),
        "proj_s_pct":       round(100 - proj_k_pct, 3),
        "ci_95_lo":         int(ci_lo_95),
        "ci_95_hi":         int(ci_hi_95),
        "ci_80_lo":         int(ci_lo_80),
        "ci_80_hi":         int(ci_hi_80),
        "sigma":            int(sigma_total),
        "ci_k_lo":          round(ci_k_lo, 3),
        "ci_k_hi":          round(ci_k_hi, 3),
    },
    "distribution": {
        "bins":    [int(x) for x in hist_centers],
        "counts":  [int(x) for x in hist_vals],
        "x_min":   int(margin_sims.min()),
        "x_max":   int(margin_sims.max()),
    }
}

out = DATA / 'forecast.json'
out.write_text(json.dumps(forecast, ensure_ascii=False, indent=2))
print(f"\n✅  Saved: {out}  ({out.stat().st_size//1024} KB)")
