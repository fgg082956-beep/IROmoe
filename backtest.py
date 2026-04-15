# backtest.py — Бэктест сигналов на исторических данных MOEX
#
# Как работает:
#   Для каждого тикера загружаем ~365 дней дневных свечей.
#   Скользящим окном (window=60 дней) прогоняем analyze_ticker() на каждой точке.
#   Фиксируем сигнал ЛОНГ/ШОРТ в день i и смотрим результат через FORWARD_DAYS.
#   Результат = (close[i+N] - close[i]) / close[i] × 100 %
#   Для лонга «успех» = результат > 0, для шорта = результат < 0.
#
#   Параллельно отслеживаем вклад каждого отдельного индикатора:
#   если индикатор сработал в нужную сторону и итоговый сигнал оказался верным — win.
#
# Вывод:
#   backtest_results.csv  — строка на каждый сигнал
#   backtest_summary.csv  — точность по тикеру + горизонту
#   indicator_stats.csv   — точность каждого индикатора
#
# Запуск:
#   python backtest.py
#   python backtest.py --tickers SBER GAZP VTBR
#   python backtest.py --days 500 --workers 6

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Импорты проекта ────────────────────────────────────────────────────────────
from data_fetcher import _get_candles_moex   # напрямую — без кэша, полная история
from analyzer import (
    calculate_rsi, calculate_macd, calculate_ema,
    calculate_bollinger_bands, calculate_atr,
    calculate_stochastic, calculate_moving_averages,
    analyze_ema_trend, analyze_obv, analyze_rsi_divergence,
    analyze_candle_patterns, analyze_volume,
    find_support_resistance,
)
from config import (
    TICKERS,
    SCORE_LONG_THRESHOLD,
    SCORE_SHORT_THRESHOLD,
    NEWS_SCORE_WEIGHT, NEWS_SCORE_CAP,
)

# ── Настройки ──────────────────────────────────────────────────────────────────
HISTORY_DAYS    = 365       # сколько дней истории грузить
MIN_WINDOW      = 60        # минимальная длина окна для анализа (нужно для EMA200)
FORWARD_DAYS    = [5, 10, 20]   # горизонты прогноза (дней вперёд)
MAX_WORKERS     = 6
REPORTS_DIR     = Path("reports")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Ядро: разбор одного сигнала ───────────────────────────────────────────────
def _indicators_snapshot(df: pd.DataFrame) -> dict:
    """
    Повторяет логику analyze_ticker(), но возвращает вклад каждого индикатора
    отдельно — без суммирования в итоговый балл.
    Нужно для indicator_stats.csv.
    """
    close = df["close"]
    rsi   = calculate_rsi(close)
    macd  = calculate_macd(close)
    ma    = calculate_moving_averages(df)
    bb    = calculate_bollinger_bands(close)
    atr_s = calculate_atr(df)
    vol   = analyze_volume(df)
    stoch = calculate_stochastic(df)
    ema_r = analyze_ema_trend(df)
    obv_r = analyze_obv(df)
    div_r = analyze_rsi_divergence(df, rsi)
    cnd_r = analyze_candle_patterns(df)

    cr  = rsi.iloc[-1]
    cm  = macd["macd"].iloc[-1];   cs  = macd["signal"].iloc[-1]
    ch  = macd["histogram"].iloc[-1]; ph = macd["histogram"].iloc[-2] if len(df) > 1 else 0
    cm5 = ma["ma5"].iloc[-1]; cm10 = ma["ma10"].iloc[-1]; cm20 = ma["ma20"].iloc[-1]
    cbp = bb["percent_b"].iloc[-1]
    ck  = stoch["k"].iloc[-1]; cd_ = stoch["d"].iloc[-1]
    pk  = stoch["k"].iloc[-2] if len(df) > 1 else 50
    pd_ = stoch["d"].iloc[-2] if len(df) > 1 else 50
    last = close.iloc[-1]

    scores = {}

    # RSI
    if not pd.isna(cr):
        if   cr < 30: scores["RSI"]  =  2
        elif cr < 40: scores["RSI"]  =  1
        elif cr > 70: scores["RSI"]  = -2
        elif cr > 60: scores["RSI"]  = -1
        else:         scores["RSI"]  =  0
    else:
        scores["RSI"] = 0

    # MACD
    if   cm > cs and ph < 0 and ch > 0: scores["MACD"] =  2
    elif cm > cs:                        scores["MACD"] =  1
    elif cm < cs and ph > 0 and ch < 0: scores["MACD"] = -2
    elif cm < cs:                        scores["MACD"] = -1
    else:                                scores["MACD"] =  0

    # EMA тренд
    scores["EMA_trend"] = ema_r["score"]

    # OBV
    scores["OBV"] = obv_r["score"]

    # RSI дивергенция
    scores["RSI_div"] = div_r["score"]

    # Свечные паттерны
    scores["Candles"] = cnd_r["score"]

    # Боллинджер
    if not pd.isna(cbp):
        if   cbp < 0:   scores["Bollinger"] =  2
        elif cbp < 0.2: scores["Bollinger"] =  1
        elif cbp > 1:   scores["Bollinger"] = -2
        elif cbp > 0.8: scores["Bollinger"] = -1
        else:           scores["Bollinger"] =  0
    else:
        scores["Bollinger"] = 0

    # Объём
    if   vol["signal"] == "бычий объём":    scores["Volume"] =  1
    elif vol["signal"] == "медвежий объём": scores["Volume"] = -1
    else:                                   scores["Volume"] =  0

    # MA выравнивание
    if   last > cm5 > cm10 > cm20: scores["MA_align"] =  1
    elif last < cm5 < cm10 < cm20: scores["MA_align"] = -1
    else:                          scores["MA_align"] =  0

    # Стохастик
    if not pd.isna(ck) and not pd.isna(cd_):
        k_up   = pk < pd_ and ck > cd_
        k_down = pk > pd_ and ck < cd_
        if   ck < 20 and k_up:   scores["Stoch"] =  2
        elif ck < 20:             scores["Stoch"] =  1
        elif ck > 80 and k_down: scores["Stoch"] = -2
        elif ck > 80:             scores["Stoch"] = -1
        else:                     scores["Stoch"] =  0
    else:
        scores["Stoch"] = 0

    return scores


def _signal_from_scores(ind_scores: dict) -> str:
    total = sum(ind_scores.values())
    if   total >= SCORE_LONG_THRESHOLD:  return "ЛОНГ"
    elif total <= SCORE_SHORT_THRESHOLD: return "ШОРТ"
    return "НЕЙТРАЛЬНО"


# ── Бэктест одного тикера ──────────────────────────────────────────────────────
def backtest_ticker(ticker: str, days: int = HISTORY_DAYS) -> list[dict]:
    """
    Возвращает список записей — по одной на каждый сигнал (не НЕЙТРАЛЬНО).
    """
    df_full = _get_candles_moex(ticker, days=days, interval=24)
    if df_full.empty or len(df_full) < MIN_WINDOW + max(FORWARD_DAYS) + 5:
        logger.warning(f"  {ticker}: недостаточно данных ({len(df_full)} свечей)")
        return []

    df_full = df_full.sort_values("begin").reset_index(drop=True)
    closes  = df_full["close"].values
    dates   = df_full["begin"].values

    records = []
    n = len(df_full)

    for i in range(MIN_WINDOW, n - max(FORWARD_DAYS)):
        window = df_full.iloc[i - MIN_WINDOW : i].copy().reset_index(drop=True)

        try:
            ind_scores = _indicators_snapshot(window)
        except Exception:
            continue

        signal = _signal_from_scores(ind_scores)
        if signal == "НЕЙТРАЛЬНО":
            continue

        entry_price = closes[i]
        if not entry_price or entry_price == 0:
            continue

        row = {
            "ticker":     ticker,
            "date":       pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
            "signal":     signal,
            "score":      sum(ind_scores.values()),
            "entry":      round(float(entry_price), 2),
        }

        # Результат через N дней
        for fwd in FORWARD_DAYS:
            future_price = closes[i + fwd]
            ret_pct = (future_price - entry_price) / entry_price * 100
            row[f"ret_{fwd}d"] = round(float(ret_pct), 2)
            # Успех: для ЛОНГ нужен рост, для ШОРТ нужно падение
            row[f"win_{fwd}d"] = (
                (signal == "ЛОНГ"  and ret_pct > 0) or
                (signal == "ШОРТ" and ret_pct < 0)
            )

        # Вклад каждого индикатора
        for ind_name, ind_score in ind_scores.items():
            row[f"ind_{ind_name}"] = ind_score

        records.append(row)

    logger.info(f"  {ticker}: {len(records)} сигналов из {n - MIN_WINDOW - max(FORWARD_DAYS)} баров")
    return records


# ── Агрегация результатов ──────────────────────────────────────────────────────
def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Точность по тикеру и горизонту."""
    rows = []
    for ticker, grp in df.groupby("ticker"):
        for sig in ["ЛОНГ", "ШОРТ", "ALL"]:
            sub = grp if sig == "ALL" else grp[grp["signal"] == sig]
            if sub.empty:
                continue
            row = {"ticker": ticker, "signal": sig, "total_signals": len(sub)}
            for fwd in FORWARD_DAYS:
                wins = sub[f"win_{fwd}d"].sum()
                row[f"win_{fwd}d"]     = int(wins)
                row[f"acc_{fwd}d_%"]   = round(wins / len(sub) * 100, 1)
                row[f"avg_ret_{fwd}d"] = round(sub[f"ret_{fwd}d"].mean(), 2)
            rows.append(row)
    return pd.DataFrame(rows)


def _build_indicator_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждого индикатора считаем:
      - сколько раз он дал ненулевой сигнал
      - в каком % случаев итоговый сигнал оказался верен (win_10d)
      - средний вклад в балл
    """
    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    rows = []
    for col in ind_cols:
        name = col[4:]   # убираем "ind_"
        active = df[df[col] != 0].copy()
        if active.empty:
            continue
        # Сигнал индикатора «бычий» если > 0
        bull_mask = active[col] > 0
        bear_mask = active[col] < 0

        for direction, mask in [("бычий (+)", bull_mask), ("медвежий (−)", bear_mask)]:
            sub = active[mask]
            if sub.empty:
                continue
            row = {
                "indicator":   name,
                "direction":   direction,
                "signals":     len(sub),
                "avg_contrib": round(sub[col].mean(), 2),
            }
            for fwd in FORWARD_DAYS:
                # Успех: бычий сигнал → нужен рост цены; медвежий → падение
                if direction == "бычий (+)":
                    wins = (sub[f"ret_{fwd}d"] > 0).sum()
                else:
                    wins = (sub[f"ret_{fwd}d"] < 0).sum()
                row[f"acc_{fwd}d_%"] = round(wins / len(sub) * 100, 1)
                row[f"avg_ret_{fwd}d"] = round(sub[f"ret_{fwd}d"].mean(), 2)
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("acc_10d_%", ascending=False)
    return out


# ── Главная функция ────────────────────────────────────────────────────────────
def run_backtest(tickers: list[str] = None, days: int = HISTORY_DAYS,
                 workers: int = MAX_WORKERS):
    if tickers is None:
        tickers = TICKERS

    logger.info("=" * 60)
    logger.info(f"Бэктест: {len(tickers)} тикеров, {days} дней истории")
    logger.info(f"Горизонты: {FORWARD_DAYS} дней  |  потоки: {workers}")
    logger.info("=" * 60)

    all_records = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(backtest_ticker, t, days): t for t in tickers}
        for future in as_completed(futures):
            records = future.result()
            all_records.extend(records)

    if not all_records:
        logger.error("Нет данных — проверь соединение с MOEX ISS")
        return

    df = pd.DataFrame(all_records)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Все сигналы
    out_signals = REPORTS_DIR / "backtest_results.csv"
    df.to_csv(out_signals, index=False, encoding="utf-8-sig")
    logger.info(f"Сигналы: {out_signals}  ({len(df)} строк)")

    # 2. Сводка по тикеру + горизонту
    df_summary = _build_summary(df)
    out_summary = REPORTS_DIR / "backtest_summary.csv"
    df_summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    logger.info(f"Сводка:  {out_summary}")

    # 3. Точность индикаторов
    df_ind = _build_indicator_stats(df)
    out_ind = REPORTS_DIR / "indicator_stats.csv"
    df_ind.to_csv(out_ind, index=False, encoding="utf-8-sig")
    logger.info(f"Индикаторы: {out_ind}")

    # ── Быстрый принт в консоль ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  СВОДКА БЭКТЕСТА — {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print("=" * 60)
    print(f"  Тикеров: {len(tickers)}   Сигналов всего: {len(df)}")
    longs  = (df["signal"] == "ЛОНГ").sum()
    shorts = (df["signal"] == "ШОРТ").sum()
    print(f"  Лонгов: {longs}   Шортов: {shorts}\n")

    # Общая точность
    for fwd in FORWARD_DAYS:
        acc = df[f"win_{fwd}d"].mean() * 100
        avg = df[f"ret_{fwd}d"].mean()
        print(f"  Горизонт +{fwd:2d}д:  точность {acc:.1f}%   ср. доход {avg:+.2f}%")

    print("\n  ТОП индикаторов (по точности на 10 дней):")
    if not df_ind.empty:
        top = df_ind.head(8)[["indicator", "direction", "signals", "acc_10d_%", "avg_ret_10d"]]
        print(top.to_string(index=False))

    print("\n  ХУДШИЕ индикаторы:")
    if not df_ind.empty:
        worst = df_ind.tail(4)[["indicator", "direction", "signals", "acc_10d_%", "avg_ret_10d"]]
        print(worst.to_string(index=False))

    print(f"\n  Файлы сохранены в {REPORTS_DIR}/")
    print("=" * 60)

    return {"signals": df, "summary": df_summary, "indicators": df_ind}


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Бэктест сигналов MOEX")
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Список тикеров (по умолчанию — все из config.py)")
    parser.add_argument("--days",    type=int,  default=HISTORY_DAYS,
                        help=f"Дней истории (по умолч. {HISTORY_DAYS})")
    parser.add_argument("--workers", type=int,  default=MAX_WORKERS,
                        help=f"Потоков (по умолч. {MAX_WORKERS})")
    args = parser.parse_args()

    run_backtest(
        tickers=args.tickers,
        days=args.days,
        workers=args.workers,
    )
