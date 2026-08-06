"""
collector.py — Snapshot del mercado P2P USDT/VES (versión GitHub Actions)
==========================================================================
Diseñado para ejecutarse SIN servidor propio: cada corrida toma UN snapshot
y lo agrega a un CSV diario (git-friendly). El workflow de Actions hace el
commit. El libro completo se guarda aparte en JSONL para re-featurizar
en el futuro sin engordar el CSV principal.

Estructura de datos:
    data/2026-08-05.csv          <- snapshots del día (una fila por corrida)
    data/books/2026-08-05.jsonl  <- libro completo por snapshot
    state/last_bcv.txt           <- última tasa BCV conocida (fallback)

Uso local (opcional):  python collector.py
"""

import csv
import datetime as dt
import json
import os
import re
import sys

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_DIR = "data"
BOOKS_DIR = os.path.join(DATA_DIR, "books")
STATE_BCV = os.path.join("state", "last_bcv.txt")

ROWS_PER_SIDE = 20
TOP_N_VWAP = 5

BINANCE_URL = "https://p2p.binance.com/bapi/c2c/v2/friendly/search/adv/search"
BCV_URL = "https://www.bcv.org.ve/"

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0 Safari/537.36"),
    "Accept": "application/json",
}

CSV_COLS = ["ts", "best_ask", "best_bid", "mid", "spread_pct",
            "vwap_ask5", "vwap_bid5", "vol_ask", "vol_bid",
            "imbalance", "bcv"]


def fetch_binance_side(trade_type: str, rows: int = ROWS_PER_SIDE):
    """BUY = anuncios donde compras USDT (asks) | SELL = donde vendes (bids)."""
    payload = {
        "page": 1, "rows": rows, "payTypes": [], "countries": [],
        "publisherType": None, "asset": "USDT",
        "tradeType": trade_type, "fiat": "VES",
    }
    r = requests.post(BINANCE_URL, headers=HEADERS, json=payload, timeout=25)
    r.raise_for_status()
    ads = []
    for item in (r.json().get("data") or []):
        adv = item.get("adv", {})
        try:
            ads.append({"price": float(adv["price"]),
                        "available": float(adv["surplusAmount"])})
        except (KeyError, TypeError, ValueError):
            continue
    return ads


def fetch_bcv():
    try:
        r = requests.get(BCV_URL, timeout=25, verify=False)
        r.raise_for_status()
        m = re.search(r'id="dolar".*?<strong>\s*([\d.,]+)', r.text, re.S)
        if m:
            return float(m.group(1).strip().replace(".", "").replace(",", "."))
    except Exception as e:
        print(f"[WARN] BCV no disponible: {e}")
    # Fallback: última tasa conocida
    try:
        with open(STATE_BCV) as f:
            return float(f.read().strip())
    except Exception:
        return None


def vwap(ads, n):
    top = ads[:n]
    tot = sum(a["available"] for a in top)
    return sum(a["price"] * a["available"] for a in top) / tot if tot else None


def main():
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    day = now.date().isoformat()

    try:
        asks = fetch_binance_side("BUY")
        bids = fetch_binance_side("SELL")
    except Exception as e:
        print(f"[ERROR] Binance no respondió: {e}")
        sys.exit(0)  # salir sin fallar el workflow; el próximo cron reintenta

    if not asks or not bids:
        print("[ERROR] Libro vacío (posible bloqueo de IP o cambio de API).")
        sys.exit(0)

    best_ask, best_bid = asks[0]["price"], bids[0]["price"]
    mid = (best_ask + best_bid) / 2
    vol_ask = sum(a["available"] for a in asks)
    vol_bid = sum(a["available"] for a in bids)
    imbalance = (vol_bid - vol_ask) / (vol_bid + vol_ask)

    bcv = fetch_bcv()
    if bcv:
        os.makedirs("state", exist_ok=True)
        with open(STATE_BCV, "w") as f:
            f.write(str(bcv))

    row = {
        "ts": now.isoformat(),
        "best_ask": best_ask, "best_bid": best_bid,
        "mid": round(mid, 6),
        "spread_pct": round((best_ask - best_bid) / mid * 100, 5),
        "vwap_ask5": round(vwap(asks, TOP_N_VWAP), 6),
        "vwap_bid5": round(vwap(bids, TOP_N_VWAP), 6),
        "vol_ask": round(vol_ask, 2), "vol_bid": round(vol_bid, 2),
        "imbalance": round(imbalance, 5),
        "bcv": bcv if bcv else "",
    }

    os.makedirs(BOOKS_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"{day}.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new_file:
            w.writeheader()
        w.writerow(row)

    with open(os.path.join(BOOKS_DIR, f"{day}.jsonl"), "a") as f:
        f.write(json.dumps({"ts": row["ts"], "asks": asks, "bids": bids},
                           separators=(",", ":")) + "\n")

    brecha = f"{(mid - bcv) / bcv * 100:.2f}%" if bcv else "n/d"
    print(f"[OK] {row['ts']}  mid={mid:.2f}  imbalance={imbalance:+.3f}  "
          f"brecha={brecha}")


if __name__ == "__main__":
    main()
