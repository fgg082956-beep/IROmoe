# data_fetcher.py — Получение данных MOEX (v5)
# [D]  Авто-ротация кэша: обрезает .parquet до 180 дней раз в неделю
# [TF] Недельные свечи через get_candles_weekly()
# [DG] Дивидендный календарь через /securities/{TICKER}/dividends.json

import time
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from config import MOEX_ISS_URL, HISTORY_DAYS, TBANK_TOKEN

logger = logging.getLogger(__name__)

CACHE_DIR      = Path("candles_cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_MAX_DAYS = 180
ROTATION_MARKER = CACHE_DIR / ".last_rotation"

USE_TBANK = bool(TBANK_TOKEN) and "СЮДА_ВСТАВЬ" not in TBANK_TOKEN

if USE_TBANK:
    try:
        from t_tech.invest import Client, CandleInterval
        from t_tech.invest.utils import quotation_to_decimal, now
    except ImportError:
        USE_TBANK = False

_tbank_cache: dict = {}


# ── Retry ─────────────────────────────────────────────────────────────────────
def _get_with_retry(url, params, retries=3, delay=1.0):
    last = None
    for attempt in range(1, retries+1):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(delay)
    raise last


# ── [D] Ротация кэша ──────────────────────────────────────────────────────────
def _rotation_needed():
    if not ROTATION_MARKER.exists():
        return True
    try:
        last = datetime.fromisoformat(ROTATION_MARKER.read_text().strip())
        return (datetime.now() - last).days >= 7
    except Exception:
        return True


def rotate_cache():
    if not _rotation_needed():
        return
    cutoff  = datetime.now() - timedelta(days=CACHE_MAX_DAYS)
    rotated = 0
    for pq in CACHE_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq)
            if "begin" not in df.columns:
                continue
            df["begin"] = pd.to_datetime(df["begin"])
            before = len(df)
            df = df[df["begin"] >= cutoff].reset_index(drop=True)
            if len(df) < before:
                df.to_parquet(pq, index=False)
                rotated += 1
        except Exception as e:
            logger.warning(f"Ротация {pq.name}: {e}")
    ROTATION_MARKER.write_text(datetime.now().isoformat())
    if rotated:
        logger.info(f"[D] Ротация: {rotated} файлов ({CACHE_MAX_DAYS} дней)")


# ── Кэш (дневные) ─────────────────────────────────────────────────────────────
def _cache_path(ticker):
    return CACHE_DIR / f"{ticker}.parquet"

def _load_cached(ticker):
    p = _cache_path(ticker)
    if p.exists():
        try: return pd.read_parquet(p)
        except Exception: pass
    return pd.DataFrame()

def _save_cache(ticker, df):
    if df.empty: return
    try: df.to_parquet(_cache_path(ticker), index=False)
    except Exception as e: logger.warning(f"Кэш {ticker}: {e}")

def _merge(old, new):
    if old.empty: return new
    if new.empty: return old
    combined = pd.concat([old, new], ignore_index=True)
    combined["begin"] = pd.to_datetime(combined["begin"])
    return combined.drop_duplicates(subset=["begin"]).sort_values("begin").reset_index(drop=True)


# ── Дневные свечи ─────────────────────────────────────────────────────────────
def get_candles(ticker: str, days: int = HISTORY_DAYS, interval: int = 24) -> pd.DataFrame:
    rotate_cache()
    cached = _load_cached(ticker)
    if not cached.empty:
        cached["begin"] = pd.to_datetime(cached["begin"])
        last_date  = cached["begin"].max().date()
        fetch_days = (date.today() - last_date).days + 1
        if fetch_days <= 1:
            cutoff = datetime.now() - timedelta(days=days)
            return cached[cached["begin"] >= cutoff].reset_index(drop=True)
        fresh  = _fetch_candles(ticker, fetch_days, interval)
        merged = _merge(cached, fresh)
        _save_cache(ticker, merged)
        cutoff = datetime.now() - timedelta(days=days)
        return merged[merged["begin"] >= cutoff].reset_index(drop=True)
    fresh = _fetch_candles(ticker, days, interval)
    _save_cache(ticker, fresh)
    return fresh


def _fetch_candles(ticker, days, interval):
    return _get_candles_tbank(ticker, days) if USE_TBANK else _get_candles_moex(ticker, days, interval)


# ── [TF] Недельные свечи ──────────────────────────────────────────────────────
def get_candles_weekly(ticker: str, weeks: int = 52) -> pd.DataFrame:
    """
    Недельные свечи (interval=7 MOEX ISS).
    weeks — глубина (по умолчанию 52 недели = ~1 год).
    Кэш не используется — запрос лёгкий (max ~52 строки).
    [TF]
    """
    days = weeks * 7
    if USE_TBANK:
        try:
            uid = _get_tbank_uid(ticker)
            if uid:
                return _get_candles_tbank_interval(
                    uid, days, CandleInterval.CANDLE_INTERVAL_WEEK)
        except Exception:
            pass
    return _get_candles_moex(ticker, days, interval=7)


def _get_candles_moex(ticker, days=HISTORY_DAYS, interval=24):
    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    date_to   = datetime.now().strftime("%Y-%m-%d")
    url       = (f"{MOEX_ISS_URL}/engines/stock/markets/shares"
                 f"/boards/TQBR/securities/{ticker}/candles.json")
    all_rows, start = [], 0
    while True:
        try:
            data = _get_with_retry(url, {"from": date_from, "till": date_to,
                                         "interval": interval, "start": start}).json()
        except Exception as e:
            logger.error(f"MOEX свечи {ticker}: {e}"); break
        rows = data.get("candles", {}).get("data", [])
        if not rows: break
        all_rows.extend(rows)
        start += len(rows)
        if len(rows) < 500: break
    if not all_rows: return pd.DataFrame()
    cols = data.get("candles", {}).get("columns", [])
    df   = pd.DataFrame(all_rows, columns=cols)
    df["begin"] = pd.to_datetime(df["begin"])
    for col in ("open", "close", "high", "low", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _get_tbank_uid(ticker):
    if ticker in _tbank_cache: return _tbank_cache[ticker]
    try:
        with Client(TBANK_TOKEN) as c:
            for inst in c.instruments.find_instrument(query=ticker).instruments:
                if inst.ticker == ticker and inst.instrument_kind.name == "INSTRUMENT_KIND_SHARE":
                    _tbank_cache[ticker] = inst.uid
                    return inst.uid
    except Exception as e:
        logger.warning(f"T-Банк uid {ticker}: {e}")
    return None


def _get_candles_tbank(ticker, days=HISTORY_DAYS):
    uid = _get_tbank_uid(ticker)
    if not uid: return _get_candles_moex(ticker, days, 24)
    return _get_candles_tbank_interval(uid, days, CandleInterval.CANDLE_INTERVAL_DAY)


def _get_candles_tbank_interval(uid, days, interval_enum):
    try:
        with Client(TBANK_TOKEN) as c:
            rows = []
            for candle in c.get_all_candles(
                instrument_id=uid,
                from_=now() - timedelta(days=days), to=now(),
                interval=interval_enum,
            ):
                if candle.is_complete:
                    rows.append({
                        "begin":  candle.time,
                        "open":   float(quotation_to_decimal(candle.open)),
                        "close":  float(quotation_to_decimal(candle.close)),
                        "high":   float(quotation_to_decimal(candle.high)),
                        "low":    float(quotation_to_decimal(candle.low)),
                        "volume": candle.volume,
                    })
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["begin"] = pd.to_datetime(df["begin"])
        return df
    except Exception as e:
        logger.warning(f"T-Банк свечи: {e}")
        return pd.DataFrame()


# ── [DG] Дивидендный календарь ────────────────────────────────────────────────
_div_cache: dict = {}   # {ticker: [(ex_date, value), ...]}

def get_dividends(ticker: str) -> list:
    """
    Возвращает список (ex_date: date, value: float) из MOEX ISS.
    ex_date = registryclosedate - 2 рабочих дня (режим T+2).
    Кэшируется на сессию.
    [DG]
    """
    if ticker in _div_cache:
        return _div_cache[ticker]
    url = f"{MOEX_ISS_URL}/securities/{ticker}/dividends.json"
    try:
        data = _get_with_retry(url, {"iss.meta": "off"}).json()
    except Exception as e:
        logger.warning(f"Дивиденды {ticker}: {e}")
        _div_cache[ticker] = []
        return []
    cols = data.get("dividends", {}).get("columns", [])
    rows = data.get("dividends", {}).get("data",    [])
    if not cols or not rows:
        _div_cache[ticker] = []
        return []

    result = []
    for row in rows:
        rec = dict(zip(cols, row))
        reg_close = rec.get("registryclosedate") or rec.get("regclosedate")
        value     = rec.get("value") or rec.get("dividend_value")
        if not reg_close or value is None:
            continue
        try:
            reg_dt  = datetime.strptime(str(reg_close)[:10], "%Y-%m-%d").date()
            # T+2: ex_date = registryclosedate - 2 рабочих дня
            ex_date = _subtract_business_days(reg_dt, 2)
            result.append({"ex_date": ex_date,
                           "reg_close": reg_dt,
                           "value": float(value)})
        except Exception:
            continue
    result.sort(key=lambda x: x["ex_date"])
    _div_cache[ticker] = result
    return result


def _subtract_business_days(d: date, n: int) -> date:
    """Вычитает n рабочих дней (пн–пт)."""
    result = d
    while n > 0:
        result -= timedelta(days=1)
        if result.weekday() < 5:   # пн=0 … пт=4
            n -= 1
    return result


def is_near_dividend_gap(ticker: str, window_days: int = 5) -> dict:
    """
    Проверяет, находится ли сегодня в зоне ±window_days дней от ex_date.
    Возвращает: {"near": bool, "days_to_ex": int|None,
                 "value": float|None, "direction": "before"|"after"|None}
    [DG]
    """
    today = date.today()
    divs  = get_dividends(ticker)
    if not divs:
        return {"near": False, "days_to_ex": None, "value": None, "direction": None}
    closest = min(divs, key=lambda x: abs((x["ex_date"] - today).days))
    delta   = (closest["ex_date"] - today).days
    if abs(delta) <= window_days:
        return {
            "near":       True,
            "days_to_ex": delta,
            "value":      closest["value"],
            "ex_date":    closest["ex_date"].isoformat(),
            "direction":  "before" if delta >= 0 else "after",
        }
    return {"near": False, "days_to_ex": delta,
            "value": closest["value"],
            "ex_date": closest["ex_date"].isoformat(),
            "direction": None}


# ── Текущая цена ──────────────────────────────────────────────────────────────
def get_current_price(ticker):
    if USE_TBANK:
        r = _get_price_tbank(ticker)
        if r: return r
    return _get_price_moex(ticker)


def _get_price_tbank(ticker):
    uid = _get_tbank_uid(ticker)
    if not uid: return None
    try:
        with Client(TBANK_TOKEN) as c:
            lp = c.market_data.get_last_prices(instrument_id=[uid])
            if not lp.last_prices: return None
            last = float(quotation_to_decimal(lp.last_prices[0].price))
            candles = list(c.get_all_candles(
                instrument_id=uid,
                from_=now()-timedelta(days=2), to=now(),
                interval=CandleInterval.CANDLE_INTERVAL_DAY,
            ))
            prev = None
            if len(candles) >= 2: prev = float(quotation_to_decimal(candles[-2].close))
            elif len(candles) == 1: prev = float(quotation_to_decimal(candles[0].open))
            chg = round((last-prev)/prev*100, 2) if prev else 0
            return {"last": last, "bid": None, "ask": None,
                    "change_pct": chg, "volume": None, "prev_close": prev}
    except Exception as e:
        logger.warning(f"T-Банк цена {ticker}: {e}")
        return None


def _get_price_moex(ticker):
    url = (f"{MOEX_ISS_URL}/engines/stock/markets/shares"
           f"/boards/TQBR/securities/{ticker}.json")
    try:
        data = _get_with_retry(url, {"iss.meta": "off",
                                     "iss.only": "marketdata,securities"}).json()
    except Exception as e:
        logger.error(f"MOEX цена {ticker}: {e}"); return None
    mdc  = data.get("marketdata", {}).get("columns", [])
    mdr  = data.get("marketdata", {}).get("data",    [])
    secc = data.get("securities",  {}).get("columns", [])
    secr = data.get("securities",  {}).get("data",    [])
    if not mdr or not secr: return None
    md   = dict(zip(mdc, mdr[0]))
    sec  = dict(zip(secc, secr[0]))
    last = md.get("LAST") or md.get("MARKETPRICE") or sec.get("PREVPRICE")
    prev = sec.get("PREVPRICE", 0)
    chg  = round((last-prev)/prev*100, 2) if prev and last else 0
    return {"last": last, "bid": md.get("BID"), "ask": md.get("OFFER"),
            "change_pct": chg, "volume": md.get("VOLTODAY"), "prev_close": prev}


def get_index_value():
    url = f"{MOEX_ISS_URL}/engines/stock/markets/index/securities/IMOEX.json"
    try:
        data = _get_with_retry(url, {"iss.meta": "off", "iss.only": "marketdata"}).json()
    except Exception as e:
        logger.error(f"IMOEX: {e}"); return None
    cols = data.get("marketdata", {}).get("columns", [])
    rows = data.get("marketdata", {}).get("data",    [])
    if not rows: return None
    md = dict(zip(cols, rows[0]))
    return {"value": md.get("CURRENTVALUE"), "change_pct": md.get("LASTCHANGEPRC")}
