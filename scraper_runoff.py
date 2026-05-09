"""
scraper_runoff.py
==================
ONPE 2026 — Segunda Vuelta Scraper
====================================
Fetches real-time runoff (segunda vuelta) results from ONPE API and
outputs data/live.json for the website.

DIFFERENCES FROM FIRST ROUND:
- Only ONE election (presidential runoff)
- Only FOUR numbers per mesa: k_votes, s_votes, blancos, nulos
- Same mesa codes as first round → perfect 1:1 table-level matching

SETUP:
  pip install curl_cffi pandas pyarrow tqdm

USAGE:
  python scraper_runoff.py                     # full run, all 92,766 mesas
  python scraper_runoff.py --update            # re-fetch pending/observed
  python scraper_runoff.py --export-only       # just export live.json from parquet
  python scraper_runoff.py --workers 8         # parallel workers
  python scraper_runoff.py --start 1 --end 500 # test subset

OUTPUT:
  data/live_raw.parquet     — raw per-mesa second round data
  data/live.json            — website-ready aggregated data

IMPORTANT: Run this in the segunda-vuelta-peru-2026/ directory.
"""

import argparse, json, os, sys, time, random, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ── Install deps ──────────────────────────────────────────────────────────────
def install(pkg):
    os.system(f"{sys.executable} -m pip install {pkg} --quiet")

for pkg in ["curl_cffi", "pandas", "pyarrow", "tqdm"]:
    try: __import__(pkg.replace('-','_'))
    except ImportError: install(pkg)

from curl_cffi import requests as cffi_requests
import pandas as pd
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_URL    = "https://resultadoelectoral.onpe.gob.pe/presentacion-backend"
DATA_DIR    = Path("data")
RAW_PATH    = DATA_DIR / "live_raw.parquet"
LIVE_PATH   = DATA_DIR / "live.json"
R1_PATH     = DATA_DIR / "r1_districts.json"
CKPT_FILE   = DATA_DIR / ".checkpoint_runoff.json"

TOTAL_MESAS = 92766
RUNOFF_ID   = 10   # Same presidencial election ID as first round

# Party codes for segunda vuelta (only 2 parties + blancos + nulos)
KEIKO_CODE   = "00000008"   # FUERZA POPULAR
SANCHEZ_CODE = "00000010"   # JUNTOS POR EL PERÚ
BLANCOS_CODE = "80"
NULOS_CODE   = "81"

FLUSH_EVERY   = 500
DELAY_MIN     = 0.05
DELAY_MAX     = 0.15
BATCH_PAUSE_N = 2000
BATCH_PAUSE_S = 12

# ── SESSION ───────────────────────────────────────────────────────────────────

def make_session():
    s = cffi_requests.Session(impersonate="chrome124")
    s.headers.update({
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
        "Referer":         "https://resultadoelectoral.onpe.gob.pe/main/actas",
        "Origin":          "https://resultadoelectoral.onpe.gob.pe",
    })
    return s

# ── FETCH ─────────────────────────────────────────────────────────────────────

def fetch_mesa(session, codigo: str, retries: int = 4):
    """Fetch one mesa. Returns parsed dict or None."""
    url = f"{BASE_URL}/actas/buscar/mesa"
    for attempt in range(retries):
        try:
            r = session.get(url, params={"codigoMesa": codigo}, timeout=20)
            if r.status_code == 204: return None
            if r.status_code == 429:
                time.sleep(60 * (attempt + 1)); continue
            if r.status_code == 503:
                time.sleep(30); continue
            r.raise_for_status()
            text = r.text.strip()
            if not text:
                if attempt < retries - 1: time.sleep(2 ** attempt); continue
                return None
            data = r.json().get("data") or []
            # Find the presidential runoff acta
            for acta in data:
                if acta.get("idEleccion") == RUNOFF_ID:
                    return acta
            return None
        except Exception as e:
            if attempt < retries - 1: time.sleep(2 ** attempt)
            else: return None

# ── PARSE ─────────────────────────────────────────────────────────────────────

def parse_acta(acta: dict) -> dict:
    """Parse a runoff presidential acta into a flat dict."""
    resultados = acta.get("resultados") or []
    k_votes = s_votes = blancos = nulos = 0
    
    for r in resultados:
        code = str(r.get("codigoPartido") or "")
        votos = r.get("votos") or 0
        if code == KEIKO_CODE:   k_votes = votos
        elif code == SANCHEZ_CODE: s_votes = votos
        elif code == BLANCOS_CODE: blancos = votos
        elif code == NULOS_CODE:   nulos = votos

    return {
        "codigo_mesa":     acta.get("codigoMesa"),
        "id_ubigeo":       acta.get("idUbigeo"),
        "region":          acta.get("ubigeoNivel01"),
        "provincia":       acta.get("ubigeoNivel02"),
        "distrito":        acta.get("ubigeoNivel03"),
        "estado":          acta.get("codigoEstadoActa"),
        "electores":       acta.get("totalElectoresHabiles") or 0,
        "votos_emitidos":  acta.get("totalVotosEmitidos") or 0,
        "votos_validos":   acta.get("totalVotosValidos") or 0,
        "k_votes":         k_votes,
        "s_votes":         s_votes,
        "blancos":         blancos,
        "nulos":           nulos,
        "codigo_local":    acta.get("codigoLocalVotacion"),
    }

# ── WORKER ────────────────────────────────────────────────────────────────────

class Worker:
    def __init__(self):
        self.session = make_session()

    def process(self, mesa_num: int) -> dict | None:
        codigo = str(mesa_num).zfill(6)
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        acta = fetch_mesa(self.session, codigo)
        if acta and acta.get("codigoEstadoActa") == "C":
            return parse_acta(acta)
        elif acta:
            # Return with pending flag — useful for tracking
            row = parse_acta(acta)
            row["k_votes"] = row["s_votes"] = row["blancos"] = row["nulos"] = None
            return row
        return None

# ── PARQUET I/O ───────────────────────────────────────────────────────────────

ckpt_lock = Lock()

def load_checkpoint():
    if CKPT_FILE.exists():
        return set(json.loads(CKPT_FILE.read_text()))
    return set()

def save_checkpoint(done):
    with ckpt_lock:
        CKPT_FILE.write_text(json.dumps(list(done)))

def append_parquet(rows, path):
    if not rows: return
    df_new = pd.DataFrame(rows)
    if path.exists():
        df_old = pd.read_parquet(path)
        # Upsert on codigo_mesa
        df_merged = pd.concat([df_old, df_new]).drop_duplicates(subset=["codigo_mesa"], keep="last")
        df_merged.to_parquet(path, index=False)
    else:
        df_new.to_parquet(path, index=False)

# ── EXPORT live.json ──────────────────────────────────────────────────────────

def export_live_json():
    """Aggregate parquet → live.json for the website."""
    if not RAW_PATH.exists():
        print("No raw data yet.")
        return

    df = pd.read_parquet(RAW_PATH)
    df_counted = df[df["estado"] == "C"].copy()

    if df_counted.empty:
        print("No counted actas yet.")
        build_empty_live()
        return

    df_counted["id_ubigeo"] = df_counted["id_ubigeo"].fillna(0).astype(int)
    df_counted["k_votes"] = pd.to_numeric(df_counted["k_votes"], errors="coerce").fillna(0)
    df_counted["s_votes"] = pd.to_numeric(df_counted["s_votes"], errors="coerce").fillna(0)
    df_counted["blancos"] = pd.to_numeric(df_counted["blancos"], errors="coerce").fillna(0)
    df_counted["nulos"]   = pd.to_numeric(df_counted["nulos"],   errors="coerce").fillna(0)
    df_counted["votos_validos"] = pd.to_numeric(df_counted["votos_validos"], errors="coerce").fillna(0)

    # National totals
    k_total  = int(df_counted["k_votes"].sum())
    s_total  = int(df_counted["s_votes"].sum())
    bl_total = int(df_counted["blancos"].sum())
    nu_total = int(df_counted["nulos"].sum())
    valid_total = int(df_counted["votos_validos"].sum())
    counted_mesas = len(df_counted)
    pct_rep = round(counted_mesas / TOTAL_MESAS * 100, 2)

    # District aggregation
    dist = df_counted.groupby("id_ubigeo").agg(
        k=("k_votes","sum"),
        s=("s_votes","sum"),
        blancos=("blancos","sum"),
        nulos=("nulos","sum"),
        cm=("codigo_mesa","count"),  # counted mesas in this district
    ).reset_index()

    # Mesa counts per district from R1 (total expected)
    r1_data = []
    if R1_PATH.exists():
        r1_data = json.loads(R1_PATH.read_text())
        r1_tm = {d["u"]: d["tm"] for d in r1_data}
    else:
        r1_tm = {}

    districts = []
    for _, row in dist.iterrows():
        ub = int(row["id_ubigeo"])
        if ub == 0: continue
        districts.append({
            "u":   ub,
            "k":   round(float(row["k"])),
            "s":   round(float(row["s"])),
            "bl":  round(float(row["blancos"])),
            "nu":  round(float(row["nulos"])),
            "cm":  int(row["cm"]),
            "tm":  r1_tm.get(ub, int(row["cm"])),
        })

    live = {
        "meta": {
            "total_mesas":   TOTAL_MESAS,
            "counted_mesas": counted_mesas,
            "pct_reported":  pct_rep,
            "k_votes":       k_total,
            "s_votes":       s_total,
            "blancos":       bl_total,
            "nulos":         nu_total,
            "valid_votes":   valid_total,
            "timestamp":     datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
            "live":          True,
        },
        "districts": districts,
    }

    LIVE_PATH.write_text(json.dumps(live, ensure_ascii=False, separators=(",", ":")))
    print(f"\n✅ live.json exported")
    print(f"   Counted: {counted_mesas:,} / {TOTAL_MESAS:,} mesas ({pct_rep:.1f}%)")
    print(f"   Keiko:   {k_total:,} ({k_total/(k_total+s_total+1)*100:.1f}%)")
    print(f"   Sánchez: {s_total:,} ({s_total/(k_total+s_total+1)*100:.1f}%)")
    print(f"   Margen:  {k_total-s_total:+,} votos")

def build_empty_live():
    live = {
        "meta": {
            "total_mesas": TOTAL_MESAS, "counted_mesas": 0, "pct_reported": 0.0,
            "k_votes": 0, "s_votes": 0, "blancos": 0, "nulos": 0,
            "valid_votes": 0, "timestamp": None, "live": False,
        },
        "districts": [],
    }
    LIVE_PATH.write_text(json.dumps(live, ensure_ascii=False, indent=2))

# ── FULL RUN ──────────────────────────────────────────────────────────────────

def run(start: int, end: int, n_workers: int):
    DATA_DIR.mkdir(exist_ok=True)
    done = load_checkpoint()
    todo = [i for i in range(start, end + 1) if str(i).zfill(6) not in done]

    print(f"\n{'='*60}")
    print(f"ONPE 2026 — Segunda Vuelta Scraper")
    print(f"Range  : {start:,} → {end:,}  ({len(todo):,} to fetch)")
    print(f"Done   : {len(done):,} in checkpoint")
    print(f"Workers: {n_workers}    Output: {DATA_DIR}")
    print(f"{'='*60}\n")

    workers = [Worker() for _ in range(n_workers)]
    buffer = []
    total_req = 0

    with tqdm(total=len(todo), unit="mesa") as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(workers[i % n_workers].process, num): num
                for i, num in enumerate(todo)
            }
            for future in as_completed(futures):
                num = futures[future]
                codigo = str(num).zfill(6)
                total_req += 1
                try:
                    row = future.result()
                except Exception as e:
                    row = None
                if row:
                    buffer.append(row)
                done.add(codigo)
                pbar.update(1)

                if len(done) % FLUSH_EVERY == 0:
                    append_parquet(buffer, RAW_PATH)
                    buffer = []
                    save_checkpoint(done)
                    export_live_json()   # Update website data

                if total_req % BATCH_PAUSE_N == 0:
                    tqdm.write(f"  Pausing {BATCH_PAUSE_S}s…")
                    time.sleep(BATCH_PAUSE_S)

    append_parquet(buffer, RAW_PATH)
    save_checkpoint(done)
    export_live_json()
    print("\n✅ Done!")

# ── UPDATE RUN (re-fetch pending mesas) ───────────────────────────────────────

def run_update(n_workers: int):
    if not RAW_PATH.exists():
        print("No data found. Run full scrape first.")
        return

    df = pd.read_parquet(RAW_PATH)
    pending = df[df["estado"] != "C"]["codigo_mesa"].tolist()
    print(f"\nRe-fetching {len(pending):,} pending mesas…")

    workers = [Worker() for _ in range(n_workers)]
    buffer = []

    with tqdm(total=len(pending), unit="mesa") as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(workers[i % n_workers].process, int(c)): c
                for i, c in enumerate(pending)
            }
            for future in as_completed(futures):
                try: row = future.result()
                except: row = None
                if row: buffer.append(row)
                pbar.update(1)

    append_parquet(buffer, RAW_PATH)
    export_live_json()
    print("\n✅ Update done!")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ONPE 2026 — Segunda Vuelta Scraper")
    p.add_argument("--start",       type=int, default=1,           help="First mesa number")
    p.add_argument("--end",         type=int, default=TOTAL_MESAS, help="Last mesa number")
    p.add_argument("--workers",     type=int, default=8,           help="Parallel workers")
    p.add_argument("--update",      action="store_true",           help="Re-fetch pending mesas")
    p.add_argument("--export-only", action="store_true",           help="Just export live.json from existing parquet")
    args = p.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if args.export_only: export_live_json()
    elif args.update:    run_update(n_workers=args.workers)
    else:                run(start=args.start, end=args.end, n_workers=args.workers)
